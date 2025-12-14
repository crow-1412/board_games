# Board Games Tutor（桌游教学助手）

> 目标：把“复杂规则”变成**可执行的教学流程**，并在教学过程中支持用户随时提问。
>
> MVP：先把 **《阿瓦隆（The Resistance: Avalon）》** 做到“从 0 教到能开局带玩/判定”的稳定体验，再抽象为可复用框架扩展到其他桌游。

---

## 核心理念

桌游教学很难靠“把规则书丢给 RAG”解决。我们将系统拆成两层：

- **规则知识层（Knowledge）**：结构化的规则单元（名词、行动、限制、例外、判定依据），可检索、可引用、可组合。
- **教学流程层（Tutor Flow）**：显式的“教学状态机/流程图”（节点、条件、检查点、练习题、常见误区）。

LLM（Qwen API）在这里的定位是：

- 把用户自然语言映射到“当前流程节点 + 规则单元”
- 在不改变流程控制逻辑的前提下，生成更自然的讲解/类比/例子
- 基于检索到的规则单元输出**带引用依据**的答案，尽量避免幻觉

---

## 产品形态（先做通用、可理解的教学形式）

推荐起步形态：**教学界面 + 小助手**（而不是纯聊天框）

- **左侧：阶段导航**（准备 / 目标 / 角色 / 回合流程 / 讨论与投票 / 结算 / 常见误区）
- **中间：步骤卡片**（每一步可勾选、带示例对话/提示、必要时提供“小测”）
- **右侧：随时提问助手**（提问会结合“你当前在第几步”来回答）
- **结构化输入优先**：例如“当前轮次”“是否已发身份”“本轮队长是谁”“已出现的否决次数”等按钮/选择器，提高判定与带玩准确度

> UI 的结构化输入是降低误判的关键；LLM 负责表达与补充，不负责控制流程。

---

## MVP 范围（阿瓦隆）

第一阶段只保证三件事：

- **新手教学模式（5–10 分钟开玩）**：setup → 目标 → 回合结构 → 关键规则 → 常见误区
- **带玩模式（向导式）**：用户选“我现在处于哪一步”，系统给出下一步该做什么/能做什么
- **裁判/判定模式（可执行结论 + 依据）**：用户输入局面（优先结构化），系统输出裁定并引用规则条款

不追求：

- 完整覆盖所有扩展/变体
- 纯聊天“全知全能”

---

## 技术架构（单仓 Monorepo，跨平台统一）

### 组件划分

- **Web 前端（教学 UI）**：Next.js/React
  - 阶段导航、步骤卡片、状态选择器
  - 右侧问答面板（带“当前进度/状态”上下文）

- **API 后端（编排与规则引擎）**：Python + FastAPI
  - 教学状态机（Tutor Flow Engine）
  - 规则检索（Knowledge Store + RAG）
  - 对话编排（Prompt/Tools/Guardrails）
  - 进度存储（SQLite）

- **数据层（本地优先）**
  - 规则知识：Markdown/JSON（可版本化）
  - 向量索引：本地轻量向量库（或先用 SQLite FTS，后续再上向量）
  - 用户进度：SQLite

- **LLM 提供方**
  - 起步：调用现成 Qwen API
  - 后续：可替换为本地 Ollama/Qwen 或自建推理服务（接口保持一致）

### 为什么这样拆

- **可控**：流程/判定可写死逻辑，避免 LLM“带偏”教学
- **可扩展**：新增桌游主要是新增 `data/<game>/` 的规则与流程文件
- **跨平台**：开发环境尽量用 Docker/Compose 固化，Mac/Windows/Linux 一致

---

## 建议目录结构（可扩展到多桌游）

> 下面是推荐结构（后续我们会逐步落地）。

```
board_games/
  apps/
    web/                # Next.js 教学界面
    api/                # FastAPI 编排服务

  packages/
    shared/             # 共享类型/协议（可选：TS 或 OpenAPI 生成）

  data/
    avalon/
      knowledge/
        glossary.md     # 名词/组件/状态定义
        rules.json      # 结构化规则单元（可引用）
        faq.md          # 常见误区/问答（教学口径）
      flow/
        flow.json       # 教学流程状态机（节点/条件/检查点/小测）
        scripts.md      # 每个节点的讲解稿（可被 LLM 润色）

  infra/
    docker/             # Dockerfile / compose（统一环境）

  env.example           # 环境变量模板（不含密钥）
  README.md
```

---

## 关键数据结构（建议）

### 1）规则单元（Rule Unit）
每条规则建议具备：

- `id`：稳定唯一 ID（用于引用）
- `title`：一句话标题
- `text`：教学版规则总结（避免直接复制整段原文）
- `tags`：如 `setup / voting / quest / fail_token / assassin`
- `conditions`：触发条件（可选）
- `exceptions`：例外/优先级（可选）
- `citations`：引用来源（如果有，尽量用“章节/页码/你自己的总结条目”，避免长段原文）

### 2）教学流程（Tutor Flow State Machine）
每个节点建议包含：

- `node_id`、`title`、`goal`（本节点的学习目标）
- `inputs`（建议 UI 采集的结构化状态）
- `steps`（步骤卡片内容）
- `checks`（是否理解的小测/判断题）
- `common_mistakes`（常见误区及纠正）
- `next`（下一节点与条件）
- `relevant_rules`（关联规则单元 id 列表）

> 这样可以做到“流程由代码控制，讲解由 LLM 表达”，并且每次回答都能带规则依据。

---

## API 设计（最小可行）

- `POST /tutor/session`：创建/加载一局教学会话
- `POST /tutor/step/advance`：基于当前节点 + 用户选择推进流程（返回步骤卡片）
- `POST /judge/query`：输入局面（结构化为主），输出裁定 + 规则引用
- `POST /chat/ask`：自由提问（会自动携带当前节点/状态）

数据协议建议用 OpenAPI 固化，前后端对齐。

---

## 本地运行与统一环境（Mac / Windows / Linux）

### 推荐：Docker/Compose 固化环境
原因：你后续要在 Windows/Linux 继续做，Docker 是成本最低的统一方案。

- 前端：Node LTS
- 后端：Python 3.11/3.12
- 数据：SQLite 文件卷

本仓库已提供最小可运行的：

- 根目录 `docker-compose.yml`
- `infra/docker/api/Dockerfile`
- `infra/docker/web/Dockerfile`

### 快速启动（Mac，本地开发）

1）准备环境变量（可选：配置后 `chat/ask` 会真实调用 Qwen）：

- 复制 `env.example` 为 `.env`，填写 `QWEN_API_KEY`

2）启动容器：

- 如果你用 Docker Desktop：直接执行 `docker compose up --build`
- 如果你用 Colima（推荐 headless）：
  - `colima start`
  - `docker-compose up --build`

3）访问：

- Web：`http://localhost:3000`
- API：`http://localhost:8000/health`

### 不推荐：只靠本地手装依赖
可以跑通，但跨平台会反复踩坑（Python/Node 版本、编译依赖、路径差异）。

---

## 密钥与配置（非常重要）

- **不要把任何 API Key 写进代码或提交到 GitHub**。
- 使用 `.env`（本地）+ `env.example`（模板）管理环境变量。
- 运行时从环境变量读取，例如：
  - `QWEN_API_KEY`（或你选择的统一命名）
  - `QWEN_BASE_URL`（如需）
  - `MODEL_NAME`（如需）

> 你刚刚给出的 Key 属于敏感信息：请立刻只保存在本地 `.env`，并避免出现在提交历史里。

---

## GitHub / Git 工作流建议（起步就养成习惯）

- 单仓：`main`（稳定）+ `dev`（日常开发）
- 所有改动走 PR（即使你一个人，也能沉淀变更记录）
- Issue 记录需求/bug（例如：阿瓦隆 flow 节点缺失、判定歧义、UI 状态不足等）
- 最小 CI：lint + test（后续再加 e2e）

> 你还没创建远程仓库时，也可以先本地初始化 git；创建 GitHub repo 后再 `git remote add origin ...`。

---

## 版权与合规提示

规则书原文通常有版权。更稳妥的做法：

- 自己写“教学版规则总结 + 示例”，不要直接分发整本规则书或大段原文
- 或仅处理用户自行上传的规则书，并限制在其本地/私域使用

---

## 接下来怎么推进（建议的最小落地顺序）

1. 初始化工程骨架（web/api/数据目录 + Docker）
2. 先把 `data/avalon/flow/flow.json` 做出 10–20 个教学节点（从开局到第一轮投票与任务）
3. 做最小前端：阶段导航 + 步骤卡片 + “我在第几步”选择
4. 后端实现：Flow Engine（状态机）+ Knowledge 检索 + Qwen 调用封装
5. 上线前再考虑服务器（先本地跑通价值）

---

## 联系与贡献

- 项目维护：`crow-1412`（GitHub）
- 建议：用 PR + Issue 方式记录讨论与决策
