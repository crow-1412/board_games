from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DATA_DIR = os.getenv("DATA_DIR", "/app/data")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/app/apps/api/var/app.db")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
).strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus").strip()


app = FastAPI(title="Board Games Tutor API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {str(r[1]) for r in cur.fetchall()}


def _ensure_sqlite() -> None:
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              game TEXT NOT NULL,
              node_id TEXT NOT NULL,
              state_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )

        cols = _sqlite_columns(conn, "sessions")
        if "state_json" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN state_json TEXT NOT NULL DEFAULT '{}' ")
        if "updated_at" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN updated_at TEXT NOT NULL DEFAULT '' ")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)")

        conn.commit()


@dataclass
class Flow:
    game: str
    schema_version: int
    start_node: str
    nodes: dict[str, dict[str, Any]]


_flow_cache: dict[str, Flow] = {}


def load_flow(game: str) -> Flow:
    if game in _flow_cache:
        return _flow_cache[game]

    path = os.path.join(DATA_DIR, game, "flow", "flow.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"flow not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load flow.json: {e}")

    if "start_node" not in raw or "nodes" not in raw:
        raise HTTPException(status_code=500, detail="invalid flow.json schema")

    flow = Flow(
        game=str(raw.get("game", game)),
        schema_version=int(raw.get("schema_version", 0)),
        start_node=str(raw["start_node"]),
        nodes=dict(raw["nodes"]),
    )
    _flow_cache[game] = flow
    return flow


class FlowOptionOut(BaseModel):
    label: str
    value: Any


class FlowFieldOut(BaseModel):
    key: str
    label: str
    type: str
    required: bool = False
    options: list[FlowOptionOut] = Field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    help: Optional[str] = None


class FlowActionOut(BaseModel):
    id: str
    label: str


class FlowNodeOut(BaseModel):
    node_id: str
    stage: str = ""
    title: str
    body: str
    fields: list[FlowFieldOut] = Field(default_factory=list)
    actions: list[FlowActionOut] = Field(default_factory=list)


def _parse_actions(node: dict[str, Any]) -> list[FlowActionOut]:
    raw_actions = node.get("actions")
    if not raw_actions:
        raw_actions = ["next"]

    out: list[FlowActionOut] = []
    for a in raw_actions:
        if isinstance(a, str):
            out.append(FlowActionOut(id=a, label=a))
        elif isinstance(a, dict):
            aid = str(a.get("id", ""))
            out.append(FlowActionOut(id=aid, label=str(a.get("label", aid))))

    # 过滤空 id
    return [x for x in out if x.id]


def _parse_fields(node: dict[str, Any]) -> list[FlowFieldOut]:
    ui = node.get("ui") or {}
    raw_fields = ui.get("fields") or []
    out: list[FlowFieldOut] = []

    for f in raw_fields:
        if not isinstance(f, dict):
            continue
        options = []
        for opt in f.get("options") or []:
            if isinstance(opt, dict) and "label" in opt and "value" in opt:
                options.append(FlowOptionOut(label=str(opt["label"]), value=opt["value"]))
        out.append(
            FlowFieldOut(
                key=str(f.get("key", "")),
                label=str(f.get("label", "")),
                type=str(f.get("type", "text")),
                required=bool(f.get("required", False)),
                options=options,
                min=f.get("min"),
                max=f.get("max"),
                help=f.get("help"),
            )
        )

    return [x for x in out if x.key]


def node_to_out(node_id: str, node: dict[str, Any]) -> FlowNodeOut:
    return FlowNodeOut(
        node_id=node_id,
        stage=str(node.get("stage", "")),
        title=str(node.get("title", "")),
        body=str(node.get("body", "")),
        fields=_parse_fields(node),
        actions=_parse_actions(node),
    )


class SessionCreateRequest(BaseModel):
    game: str = "avalon"


class SessionState(BaseModel):
    state: dict[str, Any] = Field(default_factory=dict)


class SessionCreateResponse(BaseModel):
    session_id: str
    game: str
    node: FlowNodeOut
    state: dict[str, Any]


class StepAdvanceRequest(BaseModel):
    session_id: str
    action: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class StepAdvanceResponse(BaseModel):
    session_id: str
    game: str
    node: FlowNodeOut
    state: dict[str, Any]


class ChatAskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    node_id: Optional[str] = None


class ChatAskResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)


class AgentAskRequest(BaseModel):
    question: str
    session_id: str


class AgentAskResponse(BaseModel):
    answer: str
    # 给前端用：推荐执行的动作（由当前节点的 actions 决定）
    recommended_action_id: Optional[str] = None
    recommended_action_label: Optional[str] = None
    # 给前端用：若缺信息，最多 2 个关键追问
    followup_questions: list[str] = Field(default_factory=list)
    # 不直接展示给用户：可用于日志/回放/后续“依据展示”功能
    citations: list[str] = Field(default_factory=list)


class AgentMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    created_at: str


class AgentHistoryResponse(BaseModel):
    session_id: str
    messages: list[AgentMessage] = Field(default_factory=list)


@app.on_event("startup")
def on_startup() -> None:
    _ensure_sqlite()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "time": _utc_now_iso(),
        "data_dir": DATA_DIR,
        "has_qwen_key": bool(QWEN_API_KEY),
    }


def _normalize_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    try:
        return int(str(x))
    except Exception:
        return None


def apply_derived(game: str, state: dict[str, Any]) -> dict[str, Any]:
    if game != "avalon":
        return state

    player_count = _normalize_int(state.get("player_count"))
    if player_count is not None:
        # clamp 5..10
        player_count = max(5, min(10, player_count))
        state["player_count"] = player_count

    round_no = _normalize_int(state.get("round"))
    if round_no is not None:
        round_no = max(1, min(5, round_no))
        state["round"] = round_no

    leader = _normalize_int(state.get("leader"))
    if leader is not None and player_count:
        state["leader"] = leader % player_count

    # team sizes table
    team_sizes = {
        5: [2, 3, 2, 3, 3],
        6: [2, 3, 4, 3, 4],
        7: [2, 3, 3, 4, 4],
        8: [3, 4, 4, 5, 5],
        9: [3, 4, 4, 5, 5],
        10: [3, 4, 4, 5, 5],
    }

    if player_count and round_no:
        ts = team_sizes.get(player_count)
        if ts:
            state["team_size"] = ts[round_no - 1]
        requires_two = bool(player_count >= 7 and round_no == 4)
        state["required_fail_cards"] = 2 if requires_two else 1

    # reject_count clamp 0..5（允许内部出现 5 用于判定）
    rc = _normalize_int(state.get("reject_count"))
    if rc is not None:
        state["reject_count"] = max(0, min(5, rc))

    # successes/failures clamp
    for k in ("successes", "failures"):
        v = _normalize_int(state.get(k))
        if v is not None:
            state[k] = max(0, min(5, v))

    return state


def _json_load_state(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _json_dump_state(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def _get_session(session_id: str) -> tuple[str, str, dict[str, Any]]:
    with sqlite3.connect(SQLITE_PATH) as conn:
        cur = conn.execute(
            "SELECT game, node_id, state_json FROM sessions WHERE id = ?", (session_id,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    game = str(row[0])
    node_id = str(row[1])
    state = _json_load_state(str(row[2] or "{}"))
    return game, node_id, apply_derived(game, state)


def _set_session(session_id: str, node_id: str, state: dict[str, Any]) -> None:
    now = _utc_now_iso()
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET node_id = ?, state_json = ?, updated_at = ? WHERE id = ?",
            (node_id, _json_dump_state(state), now, session_id),
        )
        conn.commit()


def _append_message(session_id: str, role: str, content: str) -> None:
    now = _utc_now_iso()
    mid = str(uuid.uuid4())
    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (mid, session_id, role, content, now),
        )
        conn.commit()


def _get_recent_messages(session_id: str, limit: int = 10) -> list[dict[str, str]]:
    lim = max(0, min(50, int(limit)))
    with sqlite3.connect(SQLITE_PATH) as conn:
        cur = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, lim),
        )
        rows = cur.fetchall()
    # 返回时间顺序（旧→新）
    rows.reverse()
    return [{"role": str(r[0]), "content": str(r[1]), "created_at": str(r[2])} for r in rows]


@app.post("/tutor/session", response_model=SessionCreateResponse)
def tutor_session(req: SessionCreateRequest) -> SessionCreateResponse:
    flow = load_flow(req.game)
    node_id = flow.start_node
    if node_id not in flow.nodes:
        raise HTTPException(status_code=500, detail="start_node missing in nodes")

    session_id = str(uuid.uuid4())
    state: dict[str, Any] = {}
    now = _utc_now_iso()

    with sqlite3.connect(SQLITE_PATH) as conn:
        conn.execute(
            "INSERT INTO sessions (id, game, node_id, state_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, req.game, node_id, _json_dump_state(state), now, now),
        )
        conn.commit()

    state = apply_derived(req.game, state)
    return SessionCreateResponse(
        session_id=session_id,
        game=req.game,
        node=node_to_out(node_id, flow.nodes[node_id]),
        state=state,
    )


@app.get("/tutor/session/{session_id}", response_model=SessionCreateResponse)
def tutor_session_get(session_id: str) -> SessionCreateResponse:
    game, node_id, state = _get_session(session_id)
    flow = load_flow(game)
    if node_id not in flow.nodes:
        raise HTTPException(status_code=500, detail="session node missing in flow")

    return SessionCreateResponse(
        session_id=session_id,
        game=game,
        node=node_to_out(node_id, flow.nodes[node_id]),
        state=state,
    )


def _resolve_value(value: Any, state: dict[str, Any]) -> Any:
    # 支持 {"$ref":"key"}
    if isinstance(value, dict) and "$ref" in value:
        return state.get(str(value.get("$ref")))
    return value


def _cmp(op: str, left: Any, right: Any) -> bool:
    op = op or "eq"

    # 尝试数字比较
    li = _normalize_int(left)
    ri = _normalize_int(right)
    if li is not None and ri is not None:
        if op == "eq":
            return li == ri
        if op == "ne":
            return li != ri
        if op == "gt":
            return li > ri
        if op == "gte":
            return li >= ri
        if op == "lt":
            return li < ri
        if op == "lte":
            return li <= ri

    if op == "in":
        if isinstance(right, (list, tuple, set)):
            return left in right
        return False

    # fallback string compare
    ls = "" if left is None else str(left)
    rs = "" if right is None else str(right)
    if op == "eq":
        return ls == rs
    if op == "ne":
        return ls != rs

    return False


def _cond_ok(cond: dict[str, Any], state: dict[str, Any]) -> bool:
    key = str(cond.get("key", ""))
    op = str(cond.get("op", "eq"))
    value = _resolve_value(cond.get("value"), state)
    return _cmp(op, state.get(key), value)


def _pick_branch(branches: list[dict[str, Any]], state: dict[str, Any]) -> str:
    for b in branches:
        if not isinstance(b, dict):
            continue
        target = b.get("then")
        if not target:
            continue
        conds = b.get("if")
        if not conds:
            return str(target)
        if isinstance(conds, list) and all(_cond_ok(c, state) for c in conds if isinstance(c, dict)):
            return str(target)
    raise HTTPException(status_code=500, detail="no branch matched")


def _resolve_to(to_spec: Any, state: dict[str, Any]) -> str:
    if isinstance(to_spec, str):
        return to_spec
    if isinstance(to_spec, dict) and "branches" in to_spec:
        branches = to_spec.get("branches")
        if isinstance(branches, list):
            return _pick_branch(branches, state)
    raise HTTPException(status_code=500, detail="invalid transition 'to' spec")


def _apply_updates(state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    # set
    raw_set = spec.get("set")
    if isinstance(raw_set, dict):
        for k, v in raw_set.items():
            state[str(k)] = v

    # inc
    raw_inc = spec.get("inc")
    if isinstance(raw_inc, dict):
        for k, dv in raw_inc.items():
            kk = str(k)
            cur = _normalize_int(state.get(kk)) or 0
            delta = _normalize_int(dv) or 0
            state[kk] = cur + delta

    return state


@app.post("/tutor/step/advance", response_model=StepAdvanceResponse)
def tutor_step_advance(req: StepAdvanceRequest) -> StepAdvanceResponse:
    game, cur_node_id, state = _get_session(req.session_id)
    flow = load_flow(game)

    cur_node = flow.nodes.get(cur_node_id)
    if not cur_node:
        raise HTTPException(status_code=500, detail="session node missing in flow")

    # merge inputs
    if req.inputs:
        for k, v in req.inputs.items():
            state[str(k)] = v

    state = apply_derived(game, state)

    action = (req.action or "next").strip()

    # transition spec priority: node.on[action] -> compatibility with node.next
    spec: dict[str, Any] = {}
    on_map = cur_node.get("on")
    if isinstance(on_map, dict) and action in on_map and isinstance(on_map[action], dict):
        spec = dict(on_map[action])
    elif action == "next" and cur_node.get("next"):
        spec = {"to": cur_node.get("next")}
    else:
        raise HTTPException(status_code=400, detail=f"unsupported action: {action}")

    # apply set/inc
    state = _apply_updates(state, spec)
    state = apply_derived(game, state)

    # 允许两种写法：
    # 1) {"to": "..."} / {"to": {"branches":[...]}}
    # 2) {"branches":[...]}  （直接作为 to-spec）
    to_spec: Any = spec.get("to") if "to" in spec else (spec if "branches" in spec else None)
    next_id = _resolve_to(to_spec, state)
    if next_id not in flow.nodes:
        raise HTTPException(status_code=500, detail=f"next node missing: {next_id}")

    _set_session(req.session_id, next_id, state)

    return StepAdvanceResponse(
        session_id=req.session_id,
        game=game,
        node=node_to_out(next_id, flow.nodes[next_id]),
        state=state,
    )


async def _call_qwen(question: str, context: str) -> str:
    if not QWEN_API_KEY:
        return (
            "（未配置 QWEN_API_KEY：当前为离线演示答案）\n"
            "我已记录你的流程节点与局面状态。你可以继续用右侧结构化输入（人数/轮次/否决次数/失败牌数）推进教学。"
        )

    sys = (
        "你是桌游教学助手，回答要简明、可执行。\n"
        "你会结合‘当前教学节点’与‘局面状态’给出下一步建议或裁定。\n"
        "尽量避免编造规则书原文，使用教学总结口径。"
    )

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"},
        ],
        "temperature": 0.2,
    }

    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(QWEN_BASE_URL, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"qwen api error: {r.status_code} {r.text}")
        data = r.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="qwen api response parse failed")


def _load_rules(game: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, game, "knowledge", "rules.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _rule_by_id(game: str, rule_id: str) -> Optional[dict[str, Any]]:
    rid = (rule_id or "").strip()
    if not rid:
        return None
    for r in _load_rules(game):
        if str(r.get("id", "")).strip() == rid:
            return r
    return None


def _rule_search(game: str, query: str, limit: int = 6) -> list[dict[str, Any]]:
    rules = _load_rules(game)
    q = (query or "").strip().lower()
    if not q:
        return []

    tokens = [t for t in "".join([c if c.isalnum() else " " for c in q]).split() if t]
    # 中文场景：额外提取高价值关键词（不依赖分词）
    zh_keywords = [
        "第4轮",
        "第四轮",
        "失败牌",
        "任务失败",
        "否决",
        "连续否决",
        "刺杀",
        "梅林",
        "派西维尔",
        "胜利条件",
        "获胜",
    ]
    kws = [k for k in zh_keywords if k in q]

    def score(rule: dict[str, Any]) -> int:
        title = str(rule.get("title", "")).lower()
        text = str(rule.get("text", "")).lower()
        s = 0
        # 粗粒度：包含就加分
        for t in tokens[:12]:
            if t in title:
                s += 6
            if t in text:
                s += 2
        # 关键词命中（对中文更友好）
        for k in kws:
            if k in title:
                s += 10
            if k in text:
                s += 4
        return s

    scored = [(score(r), r) for r in rules]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [r for s, r in scored if s > 0][: max(1, min(20, limit))]
    return out


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    # 尝试直接解析
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    # 尝试从输出中截取第一个 { ... } 块
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        sub = s[start : end + 1]
        try:
            v = json.loads(sub)
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def _is_ruling_intent(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    keywords = [
        "能不能",
        "可以不可以",
        "是否可以",
        "允不允许",
        "允许吗",
        "算不算",
        "怎么算",
        "判定",
        "裁定",
        "任务失败",
        "失败牌",
        "需要几张",
        "第4轮",
        "否决",
        "刺杀",
        "获胜",
        "胜利条件",
    ]
    return any(k in q for k in keywords)


def _is_advance_intent(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    keywords = ["下一步", "继续", "开始", "怎么做", "现在该做什么", "我现在能做什么"]
    return any(k in q for k in keywords)


def _format_rule_basis(rule_snippets: list[dict[str, Any]], max_items: int = 2) -> str:
    items = []
    for r in rule_snippets[:max_items]:
        title = str(r.get("title", "")).strip()
        text = str(r.get("text", "")).strip()
        if title and text:
            items.append(f"- {title}：{text}")
        elif title:
            items.append(f"- {title}")
    return "\n".join(items).strip()


def _deterministic_ruling_answer(game: str, question: str, state: dict[str, Any]) -> Optional[dict[str, Any]]:
    if game != "avalon":
        return None
    q = (question or "").strip()
    if not q:
        return None

    player_count = _normalize_int(state.get("player_count"))
    round_no = _normalize_int(state.get("round"))
    reject_count = _normalize_int(state.get("reject_count"))

    # 任务失败判定（最常见）
    if ("失败牌" in q) or ("任务失败" in q) or ("第4轮" in q):
        followups: list[str] = []
        if player_count is None:
            followups.append("你们这局是几个人？（5–10）")
        if round_no is None:
            followups.append("你说的是第几轮任务？（1–5）")
        if len(followups) >= 2:
            followups = followups[:2]
        if followups:
            return {
                "answer": "我可以裁定，但还差两条关键信息。\n\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(followups)]),
                "followup_questions": followups,
            }

        # 有足够信息：给通用裁定口径（失败阈值）
        requires_two = bool(player_count >= 7 and round_no == 4)
        threshold = 2 if requires_two else 1
        return {
            "answer": (
                f"结论：这轮任务需要 **{threshold} 张失败牌** 才会判定为失败。\n"
                f"- 你们是 {player_count} 人，第 {round_no} 轮。\n"
                + (
                    "- 因为是 7+ 人的第 4 轮，所以需要 2 张失败牌。"
                    if requires_two
                    else "- 按常见规则，出现 ≥1 张失败牌就算失败。"
                )
            ).strip()
        }

    # 否决累计
    if "否决" in q and ("几次" in q or "多少次" in q or "会怎样" in q or "获胜" in q):
        if reject_count is None:
            return {
                "answer": "常见规则：同一轮内累计否决达到 **5 次**，坏人直接获胜。\n\n你们目前已经否决了几次？（0–4）",
                "followup_questions": ["你们目前已经否决了几次？（0–4）"],
            }
        return {
            "answer": (
                "常见规则：同一轮内累计否决达到 **5 次**，坏人直接获胜。\n"
                f"你们目前否决次数是 {reject_count} 次。"
            )
        }

    return None


async def _agent_tutor_answer(
    game: str,
    node_id: str,
    node: dict[str, Any],
    state: dict[str, Any],
    chat_history: list[dict[str, str]],
    question: str,
) -> AgentAskResponse:
    # tools: RuleSearch + FlowState（已在入参提供）
    # 这里实现一个“受控 agent”：LLM 只负责组织表达、提出追问、建议点击哪个按钮；
    # 不能直接改 state / 跳节点。

    node_out = node_to_out(node_id, node)
    allowed_actions = [{"id": a.id, "label": a.label} for a in node_out.actions]
    allowed_fields = [{"key": f.key, "label": f.label, "type": f.type, "required": f.required} for f in node_out.fields]

    rules = _rule_search(game, question, limit=6)
    rule_snippets = [
        {
            "id": str(r.get("id", "")),
            "title": str(r.get("title", "")),
            "text": str(r.get("text", "")),
        }
        for r in rules
        if r.get("id")
    ]

    game_name_map = {
        "avalon": "阿瓦隆",
        "werewolf_standard": "狼人杀（示例板）",
        "werewolf_new_moon_12": "狼人杀：12人新月降临",
        "werewolf_dark_night_star_12": "狼人杀：暗夜星辰",
        "werewolf_awakened_guard_12": "狼人杀：月坠光渊（觉醒守卫）",
        "werewolf_awakened_witch_12": "狼人杀：觉醒女巫（毒药调配）",
        "werewolf_awakened_loner_12": "狼人杀：觉醒孤独少女（偶像机制）",
    }
    game_display_name = game_name_map.get(game, game)

    sys = (
        f"你是桌游《{game_display_name}》的教学教练（Tutor Agent）。\n"
        "你的目标是：把用户带着走完正确的流程，并在需要时做出明确裁定。\n"
        "\n"
        "硬性护栏：\n"
        "- 输出必须面向普通玩家，不要出现任何“代码/接口/变量/JSON/节点ID”等开发者信息。\n"
        "- 如果问题属于“能不能/怎么算/是否允许/判定”，必须引用至少 1 条规则依据（从给定的 rule_snippets 里选）。\n"
        "- 如果缺关键信息，只能追问 1-2 个决定性问题（followup_questions）。\n"
        "- 如果你想建议用户点击按钮，只能从 allowed_actions 里选一个。\n"
        "\n"
        "输出格式：只输出 JSON 对象（不要 Markdown），字段：\n"
        "{\n"
        "  \"answer\": string,\n"
        "  \"recommended_action_id\": string|null,\n"
        "  \"followup_questions\": string[],\n"
        "  \"citations\": string[]\n"
        "}\n"
        "其中 citations 必须是 rule_snippets 里的 id（例如 avalon.xxx 或 ww.xxx）。"
    )

    context = {
        "current_stage": node_out.stage,
        "current_title": node_out.title,
        "current_body": node_out.body,
        "chat_history": chat_history[-10:],
        "allowed_actions": allowed_actions,
        "allowed_fields": allowed_fields,
        "state": state,
        "rule_snippets": rule_snippets,
    }

    # 先做一层“确定性裁判”兜底：把最关键的判定做稳
    ruling_intent = _is_ruling_intent(question)
    advance_intent = _is_advance_intent(question)

    det = _deterministic_ruling_answer(game=game, question=question, state=state) if ruling_intent else None
    if det:
        # 为裁判回答附上依据（用户可读，不暴露 id）
        # 若搜索不到，就按场景补齐核心条目
        if not rule_snippets and game == "avalon":
            if ("失败牌" in question) or ("任务失败" in question) or ("第4轮" in question):
                for rid in [
                    "avalon.quest.fail_threshold_round4",
                    "avalon.quest.fail_threshold_default",
                ]:
                    r = _rule_by_id(game, rid)
                    if r:
                        rule_snippets.append(
                            {"id": str(r.get("id", "")), "title": str(r.get("title", "")), "text": str(r.get("text", ""))}
                        )
            elif "否决" in question:
                r = _rule_by_id(game, "avalon.voting.reject_5")
                if r:
                    rule_snippets.append(
                        {"id": str(r.get("id", "")), "title": str(r.get("title", "")), "text": str(r.get("text", ""))}
                    )

        basis = _format_rule_basis(rule_snippets, max_items=2)
        answer = str(det.get("answer", "")).strip()
        if basis:
            answer = (answer + "\n\n依据（简述）：\n" + basis).strip()

        followups = det.get("followup_questions") or []
        if not isinstance(followups, list):
            followups = []
        followups = [str(x).strip() for x in followups if str(x).strip()][:2]

        # citations 仍保留内部 id（不在前端展示）
        cite_ids = [str(r.get("id", "")).strip() for r in rule_snippets if str(r.get("id", "")).strip()][:2]
        return AgentAskResponse(
            answer=answer,
            recommended_action_id=None,
            recommended_action_label=None,
            followup_questions=followups,
            citations=cite_ids,
        )

    raw = await _call_qwen(question, json.dumps(context, ensure_ascii=False))
    obj = _extract_json_object(raw)

    if not obj or "answer" not in obj:
        # LLM 不按格式：降级为纯文本回答（仍保持用户向）
        return AgentAskResponse(answer=str(raw).strip() or "我没听清楚，你能再描述一下当前处于哪一步吗？")

    answer = str(obj.get("answer", "")).strip()
    rec_action = obj.get("recommended_action_id")
    rec_action = str(rec_action).strip() if isinstance(rec_action, str) and rec_action.strip() else None
    followups = obj.get("followup_questions") or []
    if not isinstance(followups, list):
        followups = []
    followups = [str(x).strip() for x in followups if str(x).strip()][:2]

    citations = obj.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    citations = [str(x).strip() for x in citations if str(x).strip()]

    # 约束：action 必须属于 allowed_actions
    allowed_action_ids = {a["id"] for a in allowed_actions}
    if rec_action not in allowed_action_ids:
        rec_action = None

    # 约束：citations 必须来自 rule_snippets
    allowed_cite_ids = {r["id"] for r in rule_snippets}
    citations = [c for c in citations if c in allowed_cite_ids]

    # 护栏加强：如果是裁判类问题但没给依据，则自动补齐依据（同时把依据写进答案）
    if ruling_intent and not citations and rule_snippets:
        citations = [str(rule_snippets[0].get("id", "")).strip()]
        basis = _format_rule_basis(rule_snippets, max_items=1)
        if basis:
            answer = (answer + "\n\n依据（简述）：\n" + basis).strip()

    # 轻量引导：如果用户明显想推进流程但没有建议按钮，默认推荐 next（若存在）
    if advance_intent and not rec_action and "next" in allowed_action_ids:
        rec_action = "next"

    # 生成 label（供前端按钮显示）
    label_map = {a["id"]: a["label"] for a in allowed_actions}
    rec_label = label_map.get(rec_action) if rec_action else None

    return AgentAskResponse(
        answer=answer,
        recommended_action_id=rec_action,
        recommended_action_label=rec_label,
        followup_questions=followups,
        citations=citations,
    )


@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest) -> AgentAskResponse:
    game, node_id, state = _get_session(req.session_id)
    flow = load_flow(game)
    node = flow.nodes.get(node_id)
    if not node:
        raise HTTPException(status_code=500, detail="session node missing in flow")
    state = apply_derived(game, state)
    # 读最近聊天，写入用户消息
    _append_message(req.session_id, "user", req.question)
    history = _get_recent_messages(req.session_id, limit=12)
    resp = await _agent_tutor_answer(
        game=game, node_id=node_id, node=node, state=state, chat_history=history, question=req.question
    )
    _append_message(req.session_id, "assistant", resp.answer)
    return resp


@app.get("/agent/history/{session_id}", response_model=AgentHistoryResponse)
def agent_history(session_id: str) -> AgentHistoryResponse:
    history = _get_recent_messages(session_id, limit=50)
    return AgentHistoryResponse(
        session_id=session_id,
        messages=[AgentMessage(**m) for m in history],
    )


@app.post("/chat/ask", response_model=ChatAskResponse)
async def chat_ask(req: ChatAskRequest) -> ChatAskResponse:
    context_parts: list[str] = []

    state: dict[str, Any] = {}

    if req.session_id:
        game, node_id, state = _get_session(req.session_id)
        context_parts.append(f"game={game}")
        context_parts.append(f"state={json.dumps(state, ensure_ascii=False)}")
        req.node_id = req.node_id or node_id

        if req.node_id:
            flow = load_flow(game)
            node = flow.nodes.get(req.node_id)
            if node:
                context_parts.append(f"current_node={req.node_id}:{node.get('title','')}")
                context_parts.append(str(node.get("body", "")))

    context = "\n".join([p for p in context_parts if p.strip()])
    answer = await _call_qwen(req.question, context)

    return ChatAskResponse(answer=answer, citations=[])
