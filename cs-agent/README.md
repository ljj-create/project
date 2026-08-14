# CS-Agent — 计算机科学硕博生智能体

面向计算机科学与技术研究生的 AI 学术助手，集成知识库、代码执行、论文检索、文件解析、联网搜索与飞书机器人。整体架构参考 Codex / Claude Code 的 Agent 设计，将 **会话状态、工具运行时、执行循环、模型路由** 分层解耦，便于扩展和维护。

## 核心特性

- 多模型支持：`OpenCode Go`（默认）、OpenAI、通义千问、DeepSeek、Ollama、小米 MiMo
- 多轮上下文记忆，超出窗口后自动压缩历史摘要
- RAG 知识库：算法、操作系统、网络、数据库、机器学习、科研方法、工程实践
- 原生 Function Calling 工具循环：代码执行、论文检索、文件问答、联网搜索
- 每用户独立会话状态，支持清空记忆与模型切换
- FastAPI Web API + 飞书机器人消息卡片

## 统一 Agent 架构

参考 Codex / Claude Code 的分层方式，CS-Agent 由一个小而稳定的核心契约组成：

```text
请求输入
   |
   v
CSAgent (门面 Facade)
   |-- 按 user_id 创建/获取 AgentSession
   |-- 解析模型名并委托 AgentLoop
   v
AgentLoop (执行循环)
   |-- 1. 读取 ConversationMemory 上下文
   |-- 2. 可选 RAG 检索，注入上下文
   |-- 3. LLM 原生工具调用（最多 4 轮）
   |-- 4. ToolManager 执行并回填工具结果
   |-- 5. 生成最终回答，写回 ConversationMemory
   v
LLMRouter -> OpenAILLM / QwenLLM / DeepSeekLLM / OllamaLLM / MimoLLM
```

| 层 | 文件 | 职责 |
|---|---|---|
| 门面 | `core/agent.py` | 保持稳定的 `CSAgent.chat / chat_stream / clear_memory / get_memory` API |
| 执行循环 | `core/loop.py` | 上下文构建、模型调用、工具循环、结果生成 |
| 会话状态 | `core/session.py` | 每用户会话隔离，绑定记忆与模型偏好 |
| 记忆管理 | `core/memory.py` | 短期对话轮次、长期摘要、system prompt |
| 工具运行时 | `core/tools.py` | 将 `SkillRegistry` 适配为统一工具接口 |
| 领域类型 | `core/types.py` | `AgentResult / AgentStep / AgentStatus` |
| Prompt | `core/prompt.py` | system prompt、RAG 上下文、工具结果模板 |
| 模型路由 | `llm/router.py` | `provider/model_id` 解析、实例缓存、按任务路由 |
| 模型适配 | `llm/*.py` | OpenAI 兼容、Qwen、DeepSeek、Ollama、MiMo |
| 工具实现 | `skills/*.py` | 具体 Skill 实现与注册表 |

公共调用方（`bot/handlers.py`、`scripts/run.py`）只依赖 `CSAgent`，底层循环、工具和会话实现可以独立演进。

## 项目结构

```text
cs-agent/
├── config/          # settings.py、models.yaml
├── core/            # Agent 核心分层
│   ├── agent.py     # CSAgent 门面
│   ├── loop.py      # AgentLoop 执行循环
│   ├── session.py   # 每用户会话状态
│   ├── memory.py    # 对话记忆
│   ├── tools.py     # 工具管理器
│   ├── types.py     # 统一领域类型
│   └── prompt.py    # Prompt 构建器
├── llm/             # LLM 路由与模型适配器
├── knowledge/       # RAG：loader、splitter、vectorstore、retriever
│   └── docs/        # 内置 CS 知识文档
├── skills/          # code_executor、paper_search、file_qa、web_search
├── bot/             # 飞书机器人与消息处理
├── storage/         # SQLite、ChromaDB、日志
├── scripts/         # 启动与知识库初始化脚本
└── tests/           # 单元测试
```

## 快速开始

### 1. 安装依赖

```bash
cd cs-agent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

最低配置只需一个 LLM 的 API Key。默认使用 OpenCode Go：

```env
OPENCODE_API_KEY=sk-xxx
OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
DEFAULT_MODEL=opencode-go/deepseek-v4-flash
```

### 3. 初始化知识库

```bash
python scripts/init_knowledge.py
```

### 4. 启动服务

```bash
python scripts/run.py
```

启动后：

- Web API：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- 飞书 Webhook：`http://localhost:8000/webhook/feishu`

### 5. 测试对话

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "什么是快速排序？"}'
```

## 模型配置

在 `.env` 中配置不同模型的 API Key：

```env
# OpenCode Go（默认）
OPENCODE_API_KEY=sk-xxx
OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1

# OpenAI
OPENAI_API_KEY=sk-xxx

# 通义千问
QWEN_API_KEY=sk-xxx

# DeepSeek
DEEPSEEK_API_KEY=sk-xxx

# Ollama（本地，无需 Key）
OLLAMA_BASE_URL=http://localhost:11434

# 小米 MiMo
MIMO_API_KEY=xxx
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic
```

飞书机器人内可切换模型：

```text
/model deepseek/deepseek-coder
/model qwen/qwen-plus
/models
```

## API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 对话接口 |
| `/api/models` | GET | 列出可用模型 |
| `/api/skills` | GET | 列出可用工具 |
| `/api/health` | GET | 健康检查 |
| `/webhook/feishu` | POST | 飞书 Webhook |

对话请求示例：

```json
{
  "user_id": "user123",
  "message": "解释 Transformer 的自注意力机制",
  "model": "opencode-go/deepseek-v4-flash"
}
```

## 扩展一个 Skill

1. 在 `skills/` 下继承 `BaseSkill`，实现 `execute()`。
2. 定义 `name`、`description`、`parameters`（JSON Schema）。
3. 在 `scripts/run.py` 中注册到 `SkillRegistry`。

注册后，`ToolManager` 会自动把 Skill 转换为 LLM Function Calling 定义，由 `AgentLoop` 调用。

## 运行测试

```bash
cd cs-agent
pytest tests/ -v
```

测试覆盖对话记忆、Prompt 构建、知识库拆分与 Skill 执行。

## 技术栈

| 组件 | 技术 |
|---|---|
| 后端框架 | FastAPI |
| LLM | OpenCode Go / OpenAI / 通义千问 / DeepSeek / Ollama / 小米 MiMo |
| 向量库 | ChromaDB |
| 嵌入模型 | text2vec-base-chinese |
| 数据库 | SQLite |
| 飞书机器人 | lark 开放 API（httpx 直连） |

## License

MIT
