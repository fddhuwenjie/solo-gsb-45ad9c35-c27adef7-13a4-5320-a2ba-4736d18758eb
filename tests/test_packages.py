"""可离线往返审阅包的端到端测试。

覆盖：
- 创建（冻结）：同一读事务冻结最新迁移记录 / 候选 / 完成状态 / 全部传递
  前置状态；只收录可确认、可完成或能由同包前置项解锁的条目；其余整批
  422 并按 annotation_ids[i] 定位；
- 离线往返：携带原包体与摘要回传，无冲突时在同一写事务按拓扑序追加
  确认与完成，后项可由前项解锁（unlocked_annotation_ids）；
- 冲突整包不写入：目标已被确认 / 已完成 / 新版本改变最新位置 /
  依赖图新增传递前置项（409，items[i] 定位）；
- 非法决定整批回滚：摘要不符、候选非法、包外目标、缺决定（422）；
- 幂等：同一 package_id 成功回传返回首次结果（replay=true，200），
  不重复写入；跨文档回传 422。
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.packages import PACKAGE_FORMAT


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _make_doc(client, content, title="审阅包测试"):
    r = client.post("/documents", json={"title": title, "content": content})
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def _make_ann(client, doc, text, selection, left="", right="", body="批注"):
    start = text.index(selection)
    r = client.post(
        f"/documents/{doc}/annotations",
        json={
            "body": body,
            "selection": selection,
            "left_context": left,
            "right_context": right,
            "start": start,
            "end": start + len(selection),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["annotation_id"]


def _new_version(client, doc, content):
    r = client.post(f"/documents/{doc}/versions", json={"content": content})
    assert r.status_code == 201, r.text
    return r.json()


def _add_deps(client, doc, relations):
    r = client.post(
        f"/documents/{doc}/dependencies", json={"relations": relations}
    )
    assert r.status_code == 201, r.text
    return r


def _create_package(client, doc, ids):
    return client.post(
        f"/documents/{doc}/review-packages", json={"annotation_ids": ids}
    )


def _submit(client, doc, created, decisions, package_id=None, digest=None,
            package=None):
    body = {
        "package_id": package_id or created["package_id"],
        "digest": digest or created["digest"],
        "package": package or created["package"],
        "decisions": decisions,
    }
    return client.post(f"/documents/{doc}/review-packages/submit", json=body)


def _error_codes(payload):
    return {e["error"] for e in payload["detail"]["errors"]}


def _doc_writes(doc):
    """该文档名下 confirmations / completions / review_packages 的行数。"""
    con = db.connect()
    try:
        return (
            con.execute(
                """
                SELECT COUNT(*) c FROM confirmations cc
                JOIN locations l ON l.id = cc.location_id
                JOIN annotations a ON a.id = l.annotation_id
                WHERE a.document_id = ?
                """,
                (doc,),
            ).fetchone()["c"],
            con.execute(
                "SELECT COUNT(*) c FROM completions WHERE document_id = ?", (doc,)
            ).fetchone()["c"],
            con.execute(
                "SELECT COUNT(*) c FROM review_packages WHERE document_id = ?", (doc,)
            ).fetchone()["c"],
        )
    finally:
        con.close()


V1 = "说：目标。开头。"
# 两处「目标」上下文完全相同 → 并列最高分 ambiguous
V2 = "说：目标。说：目标。"


def _setup_doc_with_ambiguity_and_chain(client):
    """doc: ann1(目标) v2 歧义；ann2(开头)；依赖 ann1→ann2。返回定位信息。"""
    doc = _make_doc(client, V1)
    a1 = _make_ann(client, doc, V1, "目标", left="说：", right="。", body="批注一")
    a2 = _make_ann(client, doc, V1, "开头", body="批注二")
    res = _new_version(client, doc, V2)
    candidates = res["results"][0]["candidates"]
    assert res["results"][0]["status"] == "ambiguous"
    _add_deps(
        client,
        doc,
        [{"prerequisite_id": a1, "dependent_id": a2}],
    )
    return doc, a1, a2, candidates


# ---------------------------------------------------------------------------
# 创建（冻结）
# ---------------------------------------------------------------------------


def test_create_package_freezes_snapshot(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    r = _create_package(client, doc, [a1, a2])
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["format"] == PACKAGE_FORMAT
    assert data["package_id"].startswith("rp_")
    assert len(data["package_id"]) == 3 + 32
    assert len(data["digest"]) == 64
    pkg = data["package"]
    assert pkg["document_id"] == doc
    assert len(pkg["items"]) == 2

    by = {i["annotation_id"]: i for i in pkg["items"]}
    i1, i2 = by[a1], by[a2]
    # 拓扑序：前置 a1 在前
    assert [i["annotation_id"] for i in pkg["items"]] == [a1, a2]
    # 冻结最新迁移记录、候选范围
    assert i1["latest"]["status"] == "ambiguous"
    assert i1["latest"]["version_number"] == 2
    assert i1["latest"]["carried_forward"] is False
    assert len(i1["latest"]["candidates"]) == 2
    assert i1["latest"]["confirmed"] is False
    # 允许的离线决定
    assert i1["actions"] == {"confirm": True, "complete": True}
    assert i2["actions"] == {"confirm": False, "complete": True}
    # 冻结全部传递前置状态
    assert i2["transitive_prerequisite_ids"] == [a1]
    assert i2["transitive_prerequisites_completed"] is False
    assert i2["requested"] is True
    assert i1["completed"] is False


def test_create_package_unique_ids(client):
    doc = _make_doc(client, V1)
    a = _make_ann(client, doc, V1, "目标", left="说：", right="。")
    _new_version(client, doc, V2)
    r1 = _create_package(client, doc, [a])
    r2 = _create_package(client, doc, [a])
    assert r1.json()["package_id"] != r2.json()["package_id"]
    assert r1.json()["digest"] == r2.json()["digest"]


def test_create_package_errors_batch(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)

    # 重复 ID + 不存在 ID：一次性 422 并按下标定位
    r = _create_package(client, doc, [a1, a1, 999])
    assert r.status_code == 422, r.text
    codes = _error_codes(r.json())
    assert codes == {"duplicate_annotation", "annotation_not_found"}
    paths = {e["path"] for e in r.json()["detail"]["errors"]}
    assert "annotation_ids[1]" in paths and "annotation_ids[2]" in paths

    # 只请求被阻塞的后续项（前置未在包内）→ blocked_by_outside_prerequisite
    r = _create_package(client, doc, [a2])
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"blocked_by_outside_prerequisite"}
    assert r.json()["detail"]["errors"][0]["path"] == "annotation_ids[0]"


def test_create_package_cross_document(client):
    d1 = _make_doc(client, V1)
    a = _make_ann(client, d1, V1, "目标", left="说：", right="。")
    d2 = _make_doc(client, V1)
    r = _create_package(client, d2, [a])
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"cross_document"}


def test_create_package_nothing_to_decide_when_completed(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    # 先完成前置 a1，再完成 a2；之后请求 a2（resolved 且已完成）→ 无决定可执行
    client.post(f"/documents/{doc}/annotations/{a1}/completions", json={})
    client.post(f"/documents/{doc}/annotations/{a2}/completions", json={})
    r = _create_package(client, doc, [a2])
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"nothing_to_decide"}


# ---------------------------------------------------------------------------
# 成功回传：确认 + 完成，按依赖顺序解锁
# ---------------------------------------------------------------------------


def test_submit_success_confirm_and_complete_chain(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    chosen = candidates[1]
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "confirm": True, "complete": True,
             "start": chosen["start"], "end": chosen["end"], "note": "取后一处"},
            {"annotation_id": a2, "complete": True, "note": "随后完成"},
        ],
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["confirmed_count"] == 1
    assert data["completed_count"] == 2
    by = {x["annotation_id"]: x for x in data["results"]}
    # 应用顺序即包内拓扑序：a1 先完成，其 unlocked 列出 a2
    assert [x["annotation_id"] for x in data["results"]] == [a1, a2]
    assert by[a1]["confirmed"] is True and by[a1]["completed"] is True
    assert by[a1]["unlocked_annotation_ids"] == [a2]
    assert by[a2]["unlocked_annotation_ids"] == []

    # 单条确认接口再确认同一歧义记录 → 409（只追加，已被包写入）
    r2 = client.post(
        f"/documents/{doc}/annotations/{a1}/confirmations",
        json={"version_number": 2, "start": chosen["start"], "end": chosen["end"]},
    )
    assert r2.status_code == 409
    # 审阅计划反映两条都已完成
    plan = client.get(f"/documents/{doc}/review-plan").json()
    assert plan["summary"]["completed"] == 2


def test_submit_confirm_only_blocked_annotation(client):
    """被阻塞的歧义批注可以只确认（确认不是完成，不解锁后续）。"""
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    chosen = candidates[0]
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "confirm": True,
             "start": chosen["start"], "end": chosen["end"]},
            {"annotation_id": a2, "complete": True},
        ],
    )
    # a1 未完成，a2 的传递前置 a1 仍未完成 → 非法决定，整批回滚
    assert r.status_code == 422, r.text
    codes = _error_codes(r.json())
    assert "blocked" in codes
    # 回滚：该文档没有任何确认 / 完成 / package 行
    assert _doc_writes(doc) == (0, 0, 0)


def test_submit_decisions_in_reverse_order_still_applied_topo(client):
    """回传决定的排列顺序无关：服务按包内拓扑序应用（后项由前项解锁）。"""
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a2, "complete": True},
            {"annotation_id": a1, "complete": True},
        ],
    )
    assert r.status_code == 201, r.text
    assert r.json()["completed_count"] == 2
    assert [x["annotation_id"] for x in r.json()["results"]] == [a1, a2]


# ---------------------------------------------------------------------------
# 冲突：整包 409 且不写入
# ---------------------------------------------------------------------------


def _freeze_then_external_confirm(client, doc, a1, a2, candidates, created):
    """冻结后用单条接口在外部确认 a1，再回传包 → location_confirmed 冲突。"""
    picked = candidates[0]
    r = client.post(
        f"/documents/{doc}/annotations/{a1}/confirmations",
        json={"version_number": 2, "start": picked["start"], "end": picked["end"]},
    )
    assert r.status_code == 201, r.text
    return _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "confirm": True, "complete": True,
             "start": candidates[1]["start"], "end": candidates[1]["end"]},
            {"annotation_id": a2, "complete": True},
        ],
    )


def test_submit_conflict_location_confirmed(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    r = _freeze_then_external_confirm(client, doc, a1, a2, candidates, created)
    assert r.status_code == 409, r.text
    codes = _error_codes(r.json())
    assert codes == {"location_confirmed"}
    err = r.json()["detail"]["errors"][0]
    assert err["path"] == "items[0]"  # 按包内条目下标定位
    assert str(a1) in err["message"]

    con = db.connect()
    try:
        # 包内决定一条都没写入（外部首条确认保留）
        assert con.execute(
            "SELECT COUNT(*) c FROM completions WHERE document_id = ?", (doc,)
        ).fetchone()["c"] == 0
        assert con.execute(
            """
            SELECT COUNT(*) c FROM confirmations c
            JOIN locations l ON l.id = c.location_id
            JOIN annotations a ON a.id = l.annotation_id
            WHERE a.document_id = ?
            """,
            (doc,),
        ).fetchone()["c"] == 1
        assert con.execute(
            "SELECT COUNT(*) c FROM review_packages WHERE document_id = ?", (doc,)
        ).fetchone()["c"] == 0
    finally:
        con.close()


def test_submit_conflict_already_completed(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    # 冻结后外部沿合法顺序完成 a1、a2；包内仍要完成 a2 → already_completed
    client.post(f"/documents/{doc}/annotations/{a1}/completions", json={})
    client.post(f"/documents/{doc}/annotations/{a2}/completions", json={})
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "complete": True},
            {"annotation_id": a2, "complete": True},
        ],
    )
    assert r.status_code == 409, r.text
    assert _error_codes(r.json()) == {"already_completed"}
    assert r.json()["detail"]["errors"][0]["path"] == "items[1]"


def test_submit_conflict_latest_location_changed_by_new_version(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    # 冻结后提交第 3 版：a1 恢复唯一落点 → resolved，最新迁移记录改变
    _new_version(client, doc, "说：目标。")
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "complete": True},
            {"annotation_id": a2, "complete": True},
        ],
    )
    assert r.status_code == 409, r.text
    assert _error_codes(r.json()) == {"latest_location_changed"}
    msg = r.json()["detail"]["errors"][0]["message"]
    assert "第 3 版" in msg
    assert _doc_writes(doc) == (0, 0, 0)


def test_submit_conflict_prerequisites_changed(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    # 冻结后新建 a3 并加边 a3→a2：a2 的传递前置集合改变
    a3 = _make_ann(client, doc, V1, "开头", body="批注三")
    _add_deps(
        client,
        doc,
        [{"prerequisite_id": a3, "dependent_id": a2}],
    )
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "complete": True},
            {"annotation_id": a2, "complete": True},
        ],
    )
    assert r.status_code == 409, r.text
    assert _error_codes(r.json()) == {"prerequisites_changed"}


# ---------------------------------------------------------------------------
# 非法决定：整批 422 回滚
# ---------------------------------------------------------------------------


def test_submit_digest_mismatch(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    r = _submit(client, doc, created, [
        {"annotation_id": a1, "complete": True},
        {"annotation_id": a2, "complete": True},
    ], digest="0" * 64)
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"digest_mismatch"}
    assert _doc_writes(doc)[2] == 0


def test_submit_tampered_package_body(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    tampered = dict(created["package"])
    tampered["created_at"] = "1999-01-01T00:00:00.000Z"
    r = _submit(client, doc, created, [
        {"annotation_id": a1, "complete": True},
        {"annotation_id": a2, "complete": True},
    ], package=tampered)
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"digest_mismatch"}


def test_submit_bad_candidate_rolls_back(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "confirm": True, "complete": True,
             "start": 40, "end": 42},
            {"annotation_id": a2, "complete": True},
        ],
    )
    assert r.status_code == 422, r.text
    assert _error_codes(r.json()) == {"bad_candidate"}
    assert _doc_writes(doc) == (0, 0, 0)


def test_submit_unknown_and_missing_decisions(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    # 含包外批注的决定
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1, "complete": True},
            {"annotation_id": a2, "complete": True},
            {"annotation_id": 999, "complete": True},
        ],
    )
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"unknown_decision"}
    # 缺少 a2 的决定
    r = _submit(
        client,
        doc,
        created,
        [{"annotation_id": a1, "complete": True}],
    )
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"missing_decision"}


def test_submit_schema_rejects_empty_decision(client):
    doc, a1, a2, _ = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    # confirm 与 complete 均为 false → schema 422
    r = _submit(
        client,
        doc,
        created,
        [
            {"annotation_id": a1},
            {"annotation_id": a2, "complete": True},
        ],
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 幂等：同一 package_id 重放
# ---------------------------------------------------------------------------


def test_submit_idempotent_replay(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    chosen = candidates[1]
    decisions = [
        {"annotation_id": a1, "confirm": True, "complete": True,
         "start": chosen["start"], "end": chosen["end"]},
        {"annotation_id": a2, "complete": True},
    ]
    r1 = _submit(client, doc, created, decisions)
    assert r1.status_code == 201
    assert r1.json()["replay"] is False
    first = r1.json()
    n_conf, n_comp, n_pkg = _doc_writes(doc)

    # 再次回传同一 package_id → 200 + replay=true，返回首次结果，不重复写入
    r2 = _submit(client, doc, created, decisions)
    assert r2.status_code == 200, r2.text
    second = r2.json()
    assert second["replay"] is True
    assert second["confirmed_count"] == first["confirmed_count"]
    assert second["completed_count"] == first["completed_count"]
    assert second["results"] == first["results"]
    assert _doc_writes(doc) == (n_conf, n_comp, n_pkg)


def test_submit_replay_to_other_document_rejected(client):
    doc, a1, a2, candidates = _setup_doc_with_ambiguity_and_chain(client)
    created = _create_package(client, doc, [a1, a2]).json()
    chosen = candidates[1]
    decisions = [
        {"annotation_id": a1, "confirm": True, "complete": True,
         "start": chosen["start"], "end": chosen["end"]},
        {"annotation_id": a2, "complete": True},
    ]
    assert _submit(client, doc, created, decisions).status_code == 201

    other = _make_doc(client, V1, title="另一篇文档")
    r = _submit(client, other, created, decisions)
    assert r.status_code == 422
    assert _error_codes(r.json()) == {"cross_document_package"}


# ---------------------------------------------------------------------------
# 冻结包外前置项的状态快照（requested=false 条目）
# ---------------------------------------------------------------------------


def test_package_includes_prerequisite_snapshot_when_already_completed(client):
    doc = _make_doc(client, V1)
    a1 = _make_ann(client, doc, V1, "目标", left="说：", right="。", body="前置")
    a2 = _make_ann(client, doc, V1, "开头", body="后续")
    _add_deps(client, doc, [{"prerequisite_id": a1, "dependent_id": a2}])
    # 前置 a1 已完成；只请求 a2：包内冻结 a1 的完成快照（requested=false）
    client.post(f"/documents/{doc}/annotations/{a1}/completions", json={})
    created = _create_package(client, doc, [a2]).json()
    by = {i["annotation_id"]: i for i in created["package"]["items"]}
    assert set(by) == {a1, a2}
    assert by[a1]["requested"] is False
    assert by[a1]["completed"] is True
    assert by[a2]["transitive_prerequisites_completed"] is True
    # 直接回传完成 a2：包外前置冻结时已完成，应用成功
    r = _submit(
        client, doc, created, [{"annotation_id": a2, "complete": True}]
    )
    assert r.status_code == 201, r.text
    assert r.json()["completed_count"] == 1
