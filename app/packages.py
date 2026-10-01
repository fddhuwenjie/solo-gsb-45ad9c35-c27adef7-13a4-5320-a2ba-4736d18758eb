"""可离线往返的审阅包：同读事务冻结、离线填写、带冲突定位的整包回传。

**创建（冻结）**——:func:`build_package` 在调用方持有的**同一个 SQLite 读事务**
（同一快照）中，为显式请求的每条批注冻结：

- 该批注**最新版本上的迁移记录**（状态、范围、匹配依据、全部候选范围、
  是否顺延占位及顺延引用、版本编号、行 ID）；
- 当时是否已有**人工确认**、是否已有**完成记录**；
- 依赖图上的**全部传递前置项**及其完成状态（前置项同样冻结最新迁移记录）。

只收录当时**可确认**（最新位置为非顺延的未确认歧义）、**可完成**（未完成）
或**能由同包前置项解锁**（其未完成的传递前置项也在包内）的条目；
无法执行任何决定（已完成且无歧义）或被包外前置项阻塞的显式请求
以带请求下标的可定位错误返回，不生成包。

包内容规范化 JSON 的 SHA-256 即内容摘要（算法与归档相同，见
:func:`app.archive.canonical_json`），任何人都可离线复算。

**回传（应用）**——:func:`apply_package` 在调用方持有的**同一个写事务**内：

1. 重新读取当前快照并逐条做冲突检查：目标已被确认
   （``location_confirmed``）、完成状态变化（``already_completed``）、
   冻结后出现了新版本导致最新迁移记录改变（``latest_location_changed``）、
   依赖图新增了前置（``prerequisites_changed``）——整包返回可定位冲突，
   **不写入任何一行**；
2. 无冲突后按**包内依赖顺序（拓扑序）**逐条应用决定：确认决定必须选中
   冻结候选之一，完成决定要求其全部传递前置项在应用到该条时已完成
   （包外前置冻结时已完成、包内前置由同包前项解锁）；任一决定非法
   （``bad_candidate`` / ``blocked`` / ``decision_not_allowed``）则
   **整包回滚**；
3. 全部决定追加（confirmations / completions 只追加）后写入 package 记录。
   同一 ``package_id`` 的重复回传直接返回首次结果，**不重复写入**。
"""
from __future__ import annotations

import heapq
import json
import secrets
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .archive import canonical_json
from .review import build_prereq_map, transitive_prerequisites

PACKAGE_FORMAT = "review-package/1"
PACKAGE_ID_PREFIX = "rp_"


class PackageError(Exception):
    """审阅包创建 / 回传发现的一条可定位错误。

    - ``error``：机器可读错误码；
    - ``message``：含业务 ID 的中文说明；
    - ``path``：定位。创建期为 ``annotation_ids[i]``；回传期为
      ``items[i]``（按包内条目下标）或 ``package`` / ``digest``。
    """

    def __init__(self, error: str, message: str, path: str):
        super().__init__(message)
        self.error = error
        self.message = message
        self.path = path

    def to_dict(self) -> Dict[str, str]:
        return {"error": self.error, "message": self.message, "path": self.path}


def new_package_id() -> str:
    """生成审阅包 ID：``rp_`` + 32 位十六进制随机串（128 位熵）。"""
    return PACKAGE_ID_PREFIX + secrets.token_hex(16)


def compute_digest(package: Dict[str, Any]) -> str:
    """包内容摘要：规范化 JSON 的 UTF-8 字节的 SHA-256（小写十六进制）。"""
    import hashlib

    return hashlib.sha256(canonical_json(package).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 快照读取（调用方须持有读事务；所有 SELECT 落在同一快照）
# ---------------------------------------------------------------------------


def _latest_locations(
    con: sqlite3.Connection, annotation_ids: Iterable[int]
) -> Dict[int, sqlite3.Row]:
    """给定批注各自**最新版本**上的迁移记录（按版本号取最高，无记录则缺席）。"""
    ids = sorted(set(annotation_ids))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"""
        SELECT l.*, v.number AS version_number
        FROM locations l
        JOIN versions v ON v.id = l.version_id
        JOIN (
            SELECT l2.annotation_id AS annotation_id, MAX(v2.number) AS max_number
            FROM locations l2
            JOIN versions v2 ON v2.id = l2.version_id
            WHERE l2.annotation_id IN ({placeholders})
            GROUP BY l2.annotation_id
        ) AS m ON m.annotation_id = l.annotation_id
        JOIN versions vv ON vv.id = l.version_id AND vv.number = m.max_number
        """,
        ids,
    ).fetchall()
    return {row["annotation_id"]: row for row in rows}


def _confirmed_locations(
    con: sqlite3.Connection, location_ids: Iterable[int]
) -> Dict[int, sqlite3.Row]:
    """迁移记录上既有的人工确认（唯一索引保证每个 location 至多一条）。"""
    ids = sorted({lid for lid in location_ids if lid is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"""
        SELECT * FROM confirmations
        WHERE location_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    return {row["location_id"]: row for row in rows}


def _completed_annotations(
    con: sqlite3.Connection, document_id: int
) -> Dict[int, sqlite3.Row]:
    """文档下全部完成记录（按批注 ID 索引）。"""
    rows = con.execute(
        "SELECT * FROM completions WHERE document_id = ?", (document_id,)
    ).fetchall()
    return {row["annotation_id"]: row for row in rows}


def _dependency_edges(
    con: sqlite3.Connection, document_id: int
) -> List[Tuple[int, int]]:
    return [
        (r["prerequisite_id"], r["dependent_id"])
        for r in con.execute(
            "SELECT prerequisite_id, dependent_id FROM dependencies "
            "WHERE document_id = ?",
            (document_id,),
        ).fetchall()
    ]


def _annotation_body(con: sqlite3.Connection, annotation_id: int) -> str:
    row = con.execute(
        "SELECT body FROM annotations WHERE id = ?", (annotation_id,)
    ).fetchone()
    return row["body"] if row is not None else ""


def _frozen_location(
    loc: Optional[sqlite3.Row], confirmation: Optional[sqlite3.Row]
) -> Dict[str, Any]:
    """冻结一条迁移记录的全部相关状态（位置 + 候选 + 确认 + 生效状态）。"""
    if loc is None:
        return {
            "location_id": None,
            "version_number": None,
            "version_id": None,
            "status": "missing",
            "start": None,
            "end": None,
            "left_match": None,
            "right_match": None,
            "score": None,
            "candidates": [],
            "carried_from": None,
            "carried_forward": False,
            "confirmed": False,
            "confirmation": None,
        }
    candidates = json.loads(loc["candidates"])
    item: Dict[str, Any] = {
        "location_id": loc["id"],
        "version_number": loc["version_number"],
        "version_id": loc["version_id"],
        "status": loc["status"],
        "start": loc["start"],
        "end": loc["end"],
        "left_match": loc["left_match"],
        "right_match": loc["right_match"],
        "score": loc["score"],
        "candidates": candidates,
        "carried_from": loc["carried_from"],
        "carried_forward": loc["carried_from"] is not None,
        "confirmed": confirmation is not None,
        "confirmation": None,
    }
    if confirmation is not None:
        item["confirmation"] = {
            "start": confirmation["start"],
            "end": confirmation["end"],
            "note": confirmation["note"],
            "created_at": confirmation["created_at"],
        }
    return item


def _package_order(
    included: Set[int], prereqs: Dict[int, Set[int]]
) -> List[int]:
    """包内条目按依赖方向的拓扑序（前置在前；同层按批注 ID 升序，确定可复现）。"""
    indegree = {aid: 0 for aid in included}
    children: Dict[int, List[int]] = {aid: [] for aid in included}
    for aid in included:
        for pre in prereqs.get(aid, ()):
            if pre in included:
                indegree[aid] += 1
                children[pre].append(aid)
    heap = [aid for aid, deg in indegree.items() if deg == 0]
    heapq.heapify(heap)
    ordered: List[int] = []
    while heap:
        aid = heapq.heappop(heap)
        ordered.append(aid)
        for child in children[aid]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    if len(ordered) != len(included):
        # 依赖边写入前已做环检测，这里防御性报错（不应发生）
        raise PackageErrorBatch(
            [
                PackageError(
                    "dependency_cycle",
                    "依赖图存在环，无法为审阅包内条目排序",
                    "items",
                )
            ]
        )
    return ordered


# ---------------------------------------------------------------------------
# 创建：同一读事务冻结
# ---------------------------------------------------------------------------


def build_package(
    con: sqlite3.Connection, document_id: int, requested_ids: List[int]
) -> Dict[str, Any]:
    """在调用方持有的读事务快照内冻结审阅包。

    假设文档存在（由调用方在同一事务内先校验）。``requested_ids`` 的
    结构问题（空 / 重复 / 不存在 / 跨文档）与业务问题（无决定可执行 /
    被包外前置项阻塞）以 :class:`PackageError` 列表一次性抛出
    （``PackageErrorBatch``），不生成包。
    """
    errors: List[PackageError] = []

    if not requested_ids:
        raise PackageErrorBatch(
            [
                PackageError(
                    "empty_selection",
                    "审阅包至少要指定一条批注",
                    "annotation_ids",
                )
            ]
        )

    # 1. 结构校验：重复 ID、批注存在性与归属（错误按下标定位）
    seen: Set[int] = set()
    request_index: Dict[int, int] = {}
    valid_requested: Set[int] = set()
    for index, aid in enumerate(requested_ids):
        path = f"annotation_ids[{index}]"
        if aid in seen:
            errors.append(
                PackageError(
                    "duplicate_annotation",
                    f"批注 {aid} 在请求中重复出现",
                    path,
                )
            )
            continue
        seen.add(aid)
        row = con.execute(
            "SELECT id, document_id FROM annotations WHERE id = ?", (aid,)
        ).fetchone()
        if row is None:
            errors.append(
                PackageError(
                    "annotation_not_found",
                    f"批注 {aid} 不存在",
                    path,
                )
            )
            continue
        if row["document_id"] != document_id:
            errors.append(
                PackageError(
                    "cross_document",
                    f"批注 {aid} 属于文档 {row['document_id']}，"
                    f"不能在文档 {document_id} 的审阅包中处理",
                    path,
                )
            )
            continue
        request_index[aid] = index
        valid_requested.add(aid)

    if errors:
        raise PackageErrorBatch(errors)

    # 2. 同一快照：依赖图、完成记录
    edges = _dependency_edges(con, document_id)
    prereqs = build_prereq_map(edges)
    completed_rows = _completed_annotations(con, document_id)
    completed = set(completed_rows)

    # 3. 可完成闭包：显式请求项未完成，且全部未完成传递前置项也在请求集合内。
    #    依赖图无环（建边时检测），记忆化 DFS 即可。
    completable: Dict[int, bool] = {}

    def can_complete(aid: int, stack: Set[int]) -> bool:
        if aid in completable:
            return completable[aid]
        if aid in completed:
            completable[aid] = False
            return False
        stack.add(aid)
        ok = True
        for pre in transitive_prerequisites(prereqs, aid):
            if pre in completed:
                continue
            if pre not in valid_requested or pre in stack:
                ok = False
                break
            if not can_complete(pre, stack):
                ok = False
                break
        stack.discard(aid)
        completable[aid] = ok
        return ok

    # 4. 冻结范围 = 显式请求项 ∪ 其全部传递前置项（含已完成 / 包外前置，
    #    回传时必须核对它们的完成状态与最新位置未变化）
    freeze_ids: Set[int] = set()
    for aid in valid_requested:
        freeze_ids.add(aid)
        freeze_ids.update(transitive_prerequisites(prereqs, aid))

    latest = _latest_locations(con, freeze_ids)
    confirmed = _confirmed_locations(
        con, [row["id"] for row in latest.values()]
    )

    def is_confirmable(aid: int) -> bool:
        """最新位置为非顺延、未确认的歧义记录（与单条确认接口的准入一致）。"""
        loc = latest.get(aid)
        if loc is None:
            return False
        if loc["status"] != "ambiguous" or loc["carried_from"] is not None:
            return False
        return loc["id"] not in confirmed

    # 5. 对每个显式请求项判定收录资格（一次性收集全部业务错误）
    included: Set[int] = set()
    for aid in sorted(valid_requested):
        path = f"annotation_ids[{request_index[aid]}]"
        loc = latest.get(aid)
        confirmable = is_confirmable(aid)
        if aid in completed:
            if confirmable:
                # 理论不可达：歧义记录与完成记录不会共存于正常流程，
                # 但冻结以库内实际状态为准，仍允许只确认
                included.add(aid)
                continue
            errors.append(
                PackageError(
                    "nothing_to_decide",
                    f"批注 {aid} 已有完成记录且最新位置无需人工确认"
                    f"（最新状态：{'无迁移记录' if loc is None else loc['status']}），"
                    "没有可离线执行的决定",
                    path,
                )
            )
            continue
        can_finish = can_complete(aid, set())
        if not confirmable and not can_finish:
            blocking = sorted(
                p
                for p in transitive_prerequisites(prereqs, aid)
                if p not in completed and p not in valid_requested
            )
            if blocking:
                errors.append(
                    PackageError(
                        "blocked_by_outside_prerequisite",
                        f"批注 {aid} 被未完成的传递前置项 "
                        f"{blocking} 阻塞，且这些前置项不在本包请求列表中；"
                        "请把它们一并加入审阅包（后项可由同包前置项解锁）",
                        path,
                    )
                )
            elif not confirmable:
                errors.append(
                    PackageError(
                        "nothing_to_decide",
                        f"批注 {aid} 最新位置无需人工确认且当前不可完成，"
                        "没有可离线执行的决定",
                        path,
                    )
                )
            else:
                errors.append(
                    PackageError(
                        "nothing_to_decide",
                        f"批注 {aid} 没有可离线执行的决定",
                        path,
                    )
                )
            continue
        included.add(aid)

    if errors:
        raise PackageErrorBatch(errors)

    # 6. 组装冻结条目（拓扑序：前置在前，回传时按此顺序应用）
    ordered = _package_order(freeze_ids, prereqs)
    created_at = con.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now') AS now"
    ).fetchone()["now"]

    items: List[Dict[str, Any]] = []
    for aid in ordered:
        loc = latest.get(aid)
        confirmation = confirmed.get(loc["id"]) if loc is not None else None
        freeze_item = _frozen_location(loc, confirmation)
        all_prereqs = sorted(transitive_prerequisites(prereqs, aid))
        items.append(
            {
                "annotation_id": aid,
                "body": _annotation_body(con, aid),
                "requested": aid in valid_requested,
                "completed": aid in completed,
                "prerequisite_ids": sorted(prereqs.get(aid, ())),
                "transitive_prerequisite_ids": all_prereqs,
                "transitive_prerequisites_completed": all(
                    p in completed for p in all_prereqs
                ),
                "actions": {
                    "confirm": aid in included and is_confirmable(aid),
                    "complete": aid in included and aid not in completed
                    and can_complete(aid, set()),
                },
                "latest": freeze_item,
            }
        )

    package: Dict[str, Any] = {
        "format": PACKAGE_FORMAT,
        "document_id": document_id,
        "created_at": created_at,
        "items": items,
    }
    return package


class PackageErrorBatch(Exception):
    """创建期收集的一批可定位错误（整批拒绝，不生成包）。"""

    def __init__(self, errors: List[PackageError]):
        super().__init__("; ".join(e.message for e in errors))
        self.errors = errors


# ---------------------------------------------------------------------------
# 回传：同写事务冲突检测 + 按依赖顺序追加
# ---------------------------------------------------------------------------


def _decision_map(decisions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(d["annotation_id"]): d for d in decisions}


def apply_package(
    con: sqlite3.Connection,
    document_id: int,
    package: Dict[str, Any],
    digest: str,
    decisions: List[Dict[str, Any]],
    package_id: str,
) -> Dict[str, Any]:
    """在调用方持有的写事务内整包回传。

    冲突（409 类）与非法决定（422 类）以 :class:`PackageErrorBatch` 抛出；
    调用方应回滚整个事务。

    幂等：同一 ``package_id`` 已有成功记录时，直接反序列化首次结果返回，
    **不做任何写入**（``replay=True``）。首次成功时在同一事务末尾追加
    ``review_packages`` 记录（调用方统一提交 / 回滚）。
    """
    # 0. 幂等：同包重复回传 → 返回首次结果，绝不重复写入
    existing = con.execute(
        "SELECT * FROM review_packages WHERE package_id = ?", (package_id,)
    ).fetchone()
    if existing is not None:
        if existing["document_id"] != document_id:
            raise PackageErrorBatch(
                [
                    PackageError(
                        "cross_document_package",
                        f"审阅包 {package_id} 属于文档 {existing['document_id']}，"
                        f"不能回传到文档 {document_id}",
                        "package",
                    )
                ]
            )
        result = json.loads(existing["result_json"])
        result["replay"] = True
        return result

    conflicts: List[PackageError] = []
    invalid: List[PackageError] = []

    # 1. 结构：格式、归属、决定目标与包内条目对齐
    if package.get("format") != PACKAGE_FORMAT:
        invalid.append(
            PackageError(
                "bad_format",
                f"审阅包格式标识应为 {PACKAGE_FORMAT!r}，"
                f"实际为 {package.get('format')!r}",
                "package",
            )
        )
    if package.get("document_id") != document_id:
        invalid.append(
            PackageError(
                "cross_document_package",
                f"审阅包属于文档 {package.get('document_id')}，"
                f"不能回传到文档 {document_id}",
                "package",
            )
        )
    if compute_digest(package) != digest.strip().lower():
        invalid.append(
            PackageError(
                "digest_mismatch",
                "审阅包内容与摘要不符：包在冻结后被改动，或摘要不是创建时所得"
                "（应为包内容规范化 JSON 的 SHA-256 小写十六进制摘要）",
                "digest",
            )
        )

    items: List[Dict[str, Any]] = package.get("items", [])
    by_annotation: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(items):
        aid = item.get("annotation_id")
        if aid in by_annotation:
            invalid.append(
                PackageError(
                    "duplicate_item",
                    f"批注 {aid} 在审阅包中重复出现",
                    f"items[{index}]",
                )
            )
            continue
        by_annotation[aid] = item

    decision_by_ann = _decision_map(decisions)
    seen_decisions: Set[int] = set()
    for index, decision in enumerate(decisions):
        aid = int(decision["annotation_id"])
        if aid in seen_decisions:
            invalid.append(
                PackageError(
                    "duplicate_decision",
                    f"批注 {aid} 出现了多条决定，每条批注至多一条",
                    f"decisions[{index}]",
                )
            )
            continue
        seen_decisions.add(aid)
        if aid not in by_annotation:
            invalid.append(
                PackageError(
                    "unknown_decision",
                    f"批注 {aid} 不在审阅包中，不能对它做决定",
                    f"decisions[{index}]",
                )
            )

    if invalid:
        raise PackageErrorBatch(invalid)

    # 2. 决定合法性（包内静态信息）：每个**可操作的收录项**恰好执行其允许的决定
    for index, item in enumerate(items):
        aid = item["annotation_id"]
        if not item.get("requested"):
            # 冻结的传递前置项：包内只能对它带决定（它必定也在 items 中、
            # requested=false 时说明它不是显式请求项——此时不允许决定）
            if aid in decision_by_ann:
                invalid.append(
                    PackageError(
                        "decision_not_allowed",
                        f"批注 {aid} 只是冻结在包中的传递前置项（未在创建时请求），"
                        "不能对它做决定；如需处理请重新创建包含它的审阅包",
                        f"items[{index}]",
                    )
                )
            continue
        actions = item.get("actions", {})
        decision = decision_by_ann.get(aid)
        if decision is None:
            invalid.append(
                PackageError(
                    "missing_decision",
                    f"批注 {aid} 在包中可执行决定，但回传未携带其决定",
                    f"items[{index}]",
                )
            )
            continue
        will_confirm = bool(decision.get("confirm"))
        will_complete = bool(decision.get("complete"))
        if not will_confirm and not will_complete:
            invalid.append(
                PackageError(
                    "missing_decision",
                    f"批注 {aid} 的决定既未确认也未完成",
                    f"decisions.{aid}",
                )
            )
        if will_confirm and not actions.get("confirm"):
            invalid.append(
                PackageError(
                    "decision_not_allowed",
                    f"批注 {aid} 冻结时的最新位置不可确认（不是非顺延的未确认歧义）",
                    f"decisions.{aid}.confirm",
                )
            )
        if will_complete and not actions.get("complete"):
            # 已完成项 / 不可完成项不允许完成决定
            invalid.append(
                PackageError(
                    "decision_not_allowed",
                    f"批注 {aid} 冻结时不可完成（已完成或被包外前置项阻塞）",
                    f"decisions.{aid}.complete",
                )
            )

    if invalid:
        raise PackageErrorBatch(invalid)

    # 3. 当前快照冲突检测（与冻结时逐一比对）
    edges_now = _dependency_edges(con, document_id)
    prereqs_now = build_prereq_map(edges_now)
    completed_now = set(_completed_annotations(con, document_id))
    latest_now = _latest_locations(con, by_annotation)
    confirmed_now = _confirmed_locations(
        con, [row["id"] for row in latest_now.values()]
    )

    item_index = {item["annotation_id"]: i for i, item in enumerate(items)}
    for aid, item in by_annotation.items():
        index = item_index[aid]
        path = f"items[{index}]"
        decision = decision_by_ann.get(aid)
        frozen = item["latest"]

        # 3.1 完成状态变化：冻结时未完成、回传时已被（包外途径）完成
        if not item["completed"] and aid in completed_now:
            if decision is not None and decision.get("complete"):
                conflicts.append(
                    PackageError(
                        "already_completed",
                        f"批注 {aid} 在包创建后已被完成，包内完成决定不能重复写入；"
                        "请基于当前状态重新创建审阅包",
                        path,
                    )
                )
            # 仅确认的决定遇到外部完成：确认仍可作用于歧义记录，不视为冲突
        elif item["completed"] and aid not in completed_now:
            # append-only 表只可能保持完成，不可能消失；发生即数据异常
            conflicts.append(
                PackageError(
                    "completion_removed",
                    f"批注 {aid} 在包创建时已完成，当前却没有完成记录"
                    "（完成记录只追加，不应消失）",
                    path,
                )
            )

        # 3.2 最新迁移记录变化：冻结后出现了新版本（location 行 ID 改变）
        loc_now = latest_now.get(aid)
        frozen_lid = frozen["location_id"]
        now_lid = loc_now["id"] if loc_now is not None else None
        if frozen_lid != now_lid:
            conflicts.append(
                PackageError(
                    "latest_location_changed",
                    f"批注 {aid} 的最新迁移记录已变化：冻结时为 location "
                    f"{frozen_lid}（第 {frozen['version_number']} 版），"
                    f"当前为 location {now_lid}"
                    f"（第 {loc_now['version_number'] if loc_now is not None else '无'} 版）；"
                    "冻结后出现了新版本，请基于当前状态重新创建审阅包",
                    path,
                )
            )
            continue  # 同一记录上的其他检查无意义

        # 3.3 目标已被确认
        if (
            loc_now is not None
            and frozen["status"] == "ambiguous"
            and not frozen["confirmed"]
            and loc_now["id"] in confirmed_now
        ):
            if decision is not None and decision.get("confirm"):
                existing_conf = confirmed_now[loc_now["id"]]
                conflicts.append(
                    PackageError(
                        "location_confirmed",
                        f"批注 {aid} 的歧义记录 location {loc_now['id']} "
                        "在包创建后已被人工确认"
                        f"（确认行 {existing_conf['id']}，范围 "
                        f"[{existing_conf['start']}, {existing_conf['end']})）；"
                        "包内确认决定不能重复写入，请基于当前状态重新创建审阅包",
                        path,
                    )
                )

        # 3.4 依赖图变化：传递前置集合必须与冻结时一致
        frozen_trans = set(item["transitive_prerequisite_ids"])
        now_trans = transitive_prerequisites(prereqs_now, aid)
        if frozen_trans != now_trans:
            added = sorted(now_trans - frozen_trans)
            removed = sorted(frozen_trans - now_trans)
            details = []
            if added:
                details.append(f"新增传递前置项 {added}")
            if removed:
                details.append(f"传递前置项 {removed} 已消失")
            conflicts.append(
                PackageError(
                    "prerequisites_changed",
                    f"批注 {aid} 的传递前置项在包创建后发生变化（{'，'.join(details)}）；"
                    "请基于当前依赖图重新创建审阅包",
                    path,
                )
            )

        # 3.5 被冻结的前置项：若它本身在包内被完成，其「完成时阻塞前提」
        # 由 3.1/3.4 已覆盖；防御性地核对冻结时已完成的包外前置仍已完成
        for pre in frozen_trans:
            if pre in by_annotation:
                continue
            if pre not in completed_now:
                conflicts.append(
                    PackageError(
                        "outside_prerequisite_not_completed",
                        f"批注 {aid} 的包外传递前置项 {pre} 在包创建时已完成，"
                        "当前却未完成（完成记录只追加，不应消失）",
                        path,
                    )
                )

    if conflicts:
        raise PackageErrorBatch(conflicts)

    # 4. 无冲突：按包内依赖顺序逐条追加（任一非法决定 → 抛异常，调用方回滚）
    #    可操作的收录项在拓扑序中的先后即「后项可由前项解锁」的顺序。
    #    解锁增量以「该完成写入前仍被阻塞、写入后全部前置已完成」为准。
    def _remaining_blockers(aid: int) -> List[int]:
        return sorted(
            p
            for p in transitive_prerequisites(prereqs_now, aid)
            if p not in completed_now and p not in newly_completed
        )

    newly_completed: Set[int] = set()
    results: List[Dict[str, Any]] = []
    confirmation_count = 0
    completion_count = 0

    for item in items:
        aid = item["annotation_id"]
        decision = decision_by_ann.get(aid)
        if decision is None:
            continue  # 仅冻结的前置项 / 无可操作项
        index = item_index[aid]
        result: Dict[str, Any] = {
            "annotation_id": aid,
            "confirmed": False,
            "completed": False,
            "confirmation_id": None,
            "completion_id": None,
            "unlocked_annotation_ids": [],
        }

        # 4.1 确认决定：候选范围必须仍是冻结候选之一
        if decision.get("confirm"):
            loc = latest_now[aid]
            candidates = json.loads(loc["candidates"])
            start = decision.get("start")
            end = decision.get("end")
            if not any(
                c["start"] == start and c["end"] == end for c in candidates
            ):
                raise PackageErrorBatch(
                    [
                        PackageError(
                            "bad_candidate",
                            f"批注 {aid} 的确认范围 [{start}, {end}) 不在冻结候选"
                            f"落点中，候选范围为 "
                            f"{json.dumps([[c['start'], c['end']] for c in candidates], ensure_ascii=False)}",
                            f"decisions.{aid}",
                        )
                    ]
                )
            cur = con.execute(
                'INSERT INTO confirmations (location_id, start, "end", note) '
                "VALUES (?, ?, ?, ?)",
                (
                    loc["id"],
                    start,
                    end,
                    decision.get("note"),
                ),
            )
            result["confirmed"] = True
            result["confirmation_id"] = cur.lastrowid
            confirmation_count += 1

        # 4.2 完成决定：此刻全部传递前置项必须已完成
        #    （包外前置冻结时已完成；包内前置已按拓扑序在前面写入）
        if decision.get("complete"):
            blocking = _remaining_blockers(aid)
            if blocking:
                raise PackageErrorBatch(
                    [
                        PackageError(
                            "blocked",
                            f"批注 {aid} 在应用时仍被未完成的传递前置项 "
                            f"{blocking} 阻塞（这些前置项未在本包中完成）",
                            f"items[{index}]",
                        )
                    ]
                )

            # 完成前仍被阻塞、本次完成后全部前置已完成的**包内待完成**后续项
            newly_unlocked: List[int] = []
            for other in items:
                oid = other["annotation_id"]
                if oid == aid or oid in newly_completed:
                    continue
                other_decision = decision_by_ann.get(oid)
                if other_decision is None or not other_decision.get("complete"):
                    continue
                before = _remaining_blockers(oid)
                if before and all(
                    p == aid or p in newly_completed for p in before
                ):
                    newly_unlocked.append(oid)

            cur = con.execute(
                "INSERT INTO completions (document_id, annotation_id, note) "
                "VALUES (?, ?, ?)",
                (document_id, aid, decision.get("note")),
            )
            result["completed"] = True
            result["completion_id"] = cur.lastrowid
            newly_completed.add(aid)
            completion_count += 1
            result["unlocked_annotation_ids"] = sorted(newly_unlocked)

        results.append(result)

    result_out = {
        "replay": False,
        "package_id": package_id,
        "document_id": document_id,
        "digest": digest.strip().lower(),
        "confirmed_count": confirmation_count,
        "completed_count": completion_count,
        "results": results,
    }

    # 5. 全部决定追加成功：写入 package 记录（与决定同一事务）
    con.execute(
        "INSERT INTO review_packages (package_id, document_id, digest, result_json)"
        " VALUES (?, ?, ?, ?)",
        (
            package_id,
            document_id,
            digest.strip().lower(),
            json.dumps(result_out, ensure_ascii=False, sort_keys=True),
        ),
    )
    return result_out
