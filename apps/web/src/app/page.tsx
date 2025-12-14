'use client';

import { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type FlowOption = { label: string; value: any };

type FlowField = {
  key: string;
  label: string;
  type: 'select' | 'number' | 'checkbox' | 'text';
  required?: boolean;
  options?: FlowOption[];
  min?: number;
  max?: number;
  help?: string | null;
};

type FlowAction = {
  id: string;
  label: string;
};

type FlowNode = {
  node_id: string;
  stage: string;
  title: string;
  body: string;
  fields: FlowField[];
  actions: FlowAction[];
};

type SessionResp = {
  session_id: string;
  game: string;
  node: FlowNode;
  state: Record<string, any>;
};

type AdvanceResp = {
  session_id: string;
  game: string;
  node: FlowNode;
  state: Record<string, any>;
};

type ChatResp = {
  answer: string;
  citations: string[];
};

type AgentResp = {
  answer: string;
  recommended_action_id?: string | null;
  recommended_action_label?: string | null;
  followup_questions?: string[];
  citations?: string[];
};

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
};

type HistoryResp = {
  session_id: string;
  messages: ChatMessage[];
};

function getApiBase(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
}

const DERIVED_KEYS = new Set(['team_size', 'required_fail_cards']);

export default function HomePage() {
  const apiBase = useMemo(() => getApiBase().replace(/\/$/, ''), []);

  const [loading, setLoading] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [node, setNode] = useState<FlowNode | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [state, setState] = useState<Record<string, any>>({});
  const [selectedGame, setSelectedGame] = useState<string>('avalon');
  const [sessionGame, setSessionGame] = useState<string>('avalon');

  const [question, setQuestion] = useState('我现在应该怎么带大家开始第一轮？');
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);
  const [agentActionId, setAgentActionId] = useState<string | null>(null);
  const [agentActionLabel, setAgentActionLabel] = useState<string | null>(null);
  const [followups, setFollowups] = useState<string[]>([]);

  async function start() {
    setLoading(true);
    setErr(null);
    setChat([]);
    setSessionId(null);
    setNode(null);
    setState({});
    // 先把当前会话桌游切过去，避免用户看到旧桌游标题/内容残留
    setSessionGame(selectedGame);
    setAgentActionId(null);
    setAgentActionLabel(null);
    setFollowups([]);
    // 根据桌游给一个更贴近场景的默认提问
    const defaultQ =
      selectedGame === 'avalon'
        ? '我现在应该怎么带大家开始第一轮？'
        : selectedGame.startsWith('werewolf_')
          ? '主持人现在该怎么开始第1夜？'
          : '我现在该怎么开始？';
    setQuestion(defaultQ);
    try {
      const r = await fetch(`${apiBase}/tutor/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game: selectedGame })
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as SessionResp;
      setSessionId(data.session_id);
      setNode(data.node);
      setState(data.state || {});
      setSessionGame(data.game || selectedGame);

      // 拉取历史（新会话一般为空，但保证刷新后可恢复）
      try {
        const hr = await fetch(`${apiBase}/agent/history/${data.session_id}`);
        if (hr.ok) {
          const hd = (await hr.json()) as HistoryResp;
          setChat(Array.isArray(hd.messages) ? hd.messages : []);
        }
      } catch {
        // ignore
      }
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function loadHistory(sid: string) {
    try {
      const hr = await fetch(`${apiBase}/agent/history/${sid}`);
      if (!hr.ok) return;
      const hd = (await hr.json()) as HistoryResp;
      setChat(Array.isArray(hd.messages) ? hd.messages : []);
    } catch {
      // ignore
    }
  }

  async function doAction(actionId: string) {
    if (!sessionId) return;
    setLoading(true);
    setErr(null);
    setAgentActionId(null);
    setAgentActionLabel(null);
    setFollowups([]);
    try {
      const r = await fetch(`${apiBase}/tutor/step/advance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, action: actionId, inputs: state })
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as AdvanceResp;
      setNode(data.node);
      setState(data.state || {});
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  async function ask() {
    if (!sessionId) return;
    setAsking(true);
    setErr(null);
    setAgentActionId(null);
    setAgentActionLabel(null);
    setFollowups([]);
    try {
      const userText = question.trim();
      if (userText) {
        setChat((c) => [...c, { role: 'user', content: userText }]);
      }
      const r = await fetch(`${apiBase}/agent/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userText, session_id: sessionId })
      });
      if (!r.ok) throw new Error(await r.text());
      const data = (await r.json()) as AgentResp;
      setChat((c) => [...c, { role: 'assistant', content: data.answer }]);
      setAgentActionId((data.recommended_action_id as string) || null);
      setAgentActionLabel((data.recommended_action_label as string) || null);
      setFollowups(Array.isArray(data.followup_questions) ? data.followup_questions : []);
      setQuestion('');
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setAsking(false);
    }
  }

  useEffect(() => {
    start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 当 sessionId 变化时恢复历史（支持刷新页面后继续对话）
  useEffect(() => {
    if (sessionId) loadHistory(sessionId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  function renderField(f: FlowField) {
    const key = f.key;
    const value = state?.[key];

    const readOnly = DERIVED_KEYS.has(key);

    if (f.type === 'checkbox') {
      return (
        <label key={key} className="small" style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <input
            type="checkbox"
            checked={!!value}
            disabled={readOnly}
            onChange={(e) => setState((s) => ({ ...s, [key]: e.target.checked }))}
          />
          <span>
            <b>{f.label}</b>
            {f.required ? '（必填）' : ''}
          </span>
        </label>
      );
    }

    if (f.type === 'select') {
      const opts = f.options || [];
      const v = value ?? opts?.[0]?.value ?? '';
      return (
        <div key={key}>
          <div className="small">
            <b>{f.label}</b>
            {f.required ? '（必填）' : ''}
          </div>
          <select
            className="input"
            value={String(v)}
            disabled={readOnly}
            onChange={(e) => {
              const raw = e.target.value;
              // try number
              const asNumber = Number(raw);
              const nextValue = Number.isNaN(asNumber) ? raw : asNumber;
              setState((s) => ({ ...s, [key]: nextValue }));
            }}
          >
            {opts.map((o) => (
              <option key={`${key}-${String(o.value)}`} value={String(o.value)}>
                {o.label}
              </option>
            ))}
          </select>
          {f.help ? <div className="small">{f.help}</div> : null}
        </div>
      );
    }

    if (f.type === 'number') {
      return (
        <div key={key}>
          <div className="small">
            <b>{f.label}</b>
            {f.required ? '（必填）' : ''}
          </div>
          <input
            className="input"
            type="number"
            value={value ?? ''}
            min={typeof f.min === 'number' ? f.min : undefined}
            max={typeof f.max === 'number' ? f.max : undefined}
            readOnly={readOnly}
            onChange={(e) => {
              const raw = e.target.value;
              setState((s) => ({ ...s, [key]: raw === '' ? '' : Number(raw) }));
            }}
          />
          {f.help ? <div className="small">{f.help}</div> : null}
        </div>
      );
    }

    // text
    return (
      <div key={key}>
        <div className="small">
          <b>{f.label}</b>
          {f.required ? '（必填）' : ''}
        </div>
        <input
          className="input"
          value={value ?? ''}
          readOnly={readOnly}
          onChange={(e) => setState((s) => ({ ...s, [key]: e.target.value }))}
        />
        {f.help ? <div className="small">{f.help}</div> : null}
      </div>
    );
  }

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: 0.2 }}>桌游教学</div>
          <div className="small">
            <span style={{ marginRight: 10 }}>当前桌游：</span>
            <select
              className="input"
              style={{ width: 220, display: 'inline-block', padding: '8px 10px', borderRadius: 12 }}
              value={selectedGame}
              onChange={(e) => setSelectedGame(e.target.value as any)}
              disabled={loading}
            >
              <option value="avalon">阿瓦隆</option>
              <option value="werewolf_standard">狼人杀（标准示例）</option>
              <option value="werewolf_new_moon_12">狼人杀：12人新月降临</option>
              <option value="werewolf_dark_night_star_12">狼人杀：暗夜星辰</option>
              <option value="werewolf_awakened_guard_12">狼人杀：月坠光渊（觉醒守卫）</option>
              <option value="werewolf_awakened_witch_12">狼人杀：觉醒女巫（毒药调配）</option>
              <option value="werewolf_awakened_loner_12">狼人杀：觉醒孤独少女（偶像机制）</option>
            </select>
          </div>
        </div>
        <button className="button" onClick={start} disabled={loading} style={{ background: '#0f172a', boxShadow: 'none' }}>
          重新开始
        </button>
      </div>

      <div className="row">
      <div className="card">
        <div className="h1">
          {sessionGame === 'avalon'
            ? '阿瓦隆教学助手'
            : sessionGame.startsWith('werewolf_')
              ? '狼人杀教学助手'
              : '桌游教学助手'}
        </div>
        <hr className="hr" />

        {err ? (
          <div className="small" style={{ color: '#b91c1c' }}>
            出现了一点问题，请点击“新开会话”重试。
          </div>
        ) : null}

        {node ? (
          <>
            <div className="small">
              {node.stage ? <b>{node.stage}</b> : null}
            </div>
            <h2 style={{ margin: '10px 0 8px', fontSize: 18 }}>{node.title}</h2>
            <div className="md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{node.body}</ReactMarkdown>
            </div>
            <hr className="hr" />
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {(node.actions || []).map((a) => (
                <button key={a.id} className="button" onClick={() => doAction(a.id)} disabled={loading}>
                  {a.label}
                </button>
              ))}
              <button
                className="button"
                onClick={start}
                disabled={loading}
                style={{ background: '#ffffff', color: '#0f172a', borderColor: 'rgba(15,23,42,0.18)', boxShadow: 'none' }}
              >
                新开会话
              </button>
            </div>
          </>
        ) : (
          <div className="small">{loading ? '加载中...' : '未加载到教学节点'}</div>
        )}
      </div>

      <div className="card">
        <div className="h1">当前信息</div>
        <div className="small">
          你可以在这里补充当前局面，助手会据此给出更准确的下一步提示与结算。
        </div>
        <hr className="hr" />

        {node?.fields?.length ? (
          <div style={{ display: 'grid', gap: 10 }}>{node.fields.map(renderField)}</div>
        ) : (
          <div className="small">当前节点不需要额外输入。</div>
        )}

        <hr className="hr" />

        <div className="h1">随时提问</div>
        <div className="small">
          直接用自然语言描述你们现在的情况或疑问即可。
        </div>
        <hr className="hr" />

        <div className="chat" aria-label="聊天记录">
          {chat.length ? (
            chat.map((m, idx) => (
              <div key={idx} className={`bubble ${m.role}`}>
                <div className="md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                </div>
              </div>
            ))
          ) : (
            <div className="small">还没有对话记录。你可以从“现在该做什么？”开始。</div>
          )}
        </div>

        <div style={{ height: 12 }} />
        <textarea
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={5}
          placeholder="输入你的问题…"
        />
        <div style={{ height: 10 }} />
        <button className="button" onClick={ask} disabled={asking || !sessionId}>
          {asking ? '发送中...' : '发送'}
        </button>
        {agentActionId && agentActionLabel ? (
          <>
            <div style={{ height: 10 }} />
            <button
              className="button"
              onClick={() => doAction(agentActionId)}
              disabled={loading || !sessionId}
              style={{ background: '#16a34a', boxShadow: '0 8px 18px rgba(22,163,74,0.18)' }}
            >
              按建议执行：{agentActionLabel}
            </button>
          </>
        ) : null}
        {followups.length ? (
          <>
            <hr className="hr" />
            <div className="small">
              <b>为了更准确，我还想确认：</b>
            </div>
            <div className="pre">{followups.map((q, i) => `${i + 1}. ${q}`).join('\n')}</div>
          </>
        ) : null}
      </div>
    </div>
    </>
  );
}
