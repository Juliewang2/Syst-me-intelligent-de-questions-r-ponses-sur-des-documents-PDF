# 📚 PDF Chat：基于混合检索的 RAG 文档问答系统

上传 PDF，像和 ChatGPT 聊天一样提问；回答**只依据文档内容**，并标注出自哪份文档的第几页。

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-dense%20retrieval-4267B2)
![BM25](https://img.shields.io/badge/BM25-sparse%20retrieval-orange)
![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**评估结果（449 页教材 · 40 题测试集）：**相比纯向量检索基线，
**hit@4 0.81 → 0.94**，**MRR 0.70 → 0.88**，**答案正确性 0.78 → 0.93**，**忠实度 0.96 → 1.00**。
详见 [效果评估](#效果评估)。

---

## 目录

1. [一句话理解这个系统](#一句话理解这个系统)
2. [功能](#功能)
3. [两条核心流程](#两条核心流程)
4. [在基础版上做的优化](#在基础版上做的优化)
5. [效果评估](#效果评估)
6. [技术栈](#技术栈)
7. [项目结构](#项目结构)
8. [快速开始](#快速开始)
9. [配置项](#配置项)
10. [API](#api)
11. [测试](#测试)
12. [已知局限与后续方向](#已知局限与后续方向)

---

## 一句话理解这个系统

大模型（LLM）像一位**博学但没读过你这份文件的教授**：直接问它"我这份合同第 5 条写了什么"，
它不知道，还可能编一个答案。

**RAG（检索增强生成）**的做法是：先派一个**"图书管理员"**到你的 PDF 里找出最相关的几段话，
再把这几段话连同问题一起交给教授，让他**只根据这几段话回答**。

本项目就是在实现"图书管理员 + 教授"这个流程，并把"图书管理员"做得更准：
**向量检索懂意思，BM25 认关键词，重排模型再精挑细选。**

---

## 功能

- **多 PDF 上传与管理**：拖拽上传，支持在所有文档中提问，也可以只针对某一份文档提问
- **扫描版 PDF**：没有文字层的页面自动 OCR（中英文）
- **混合检索**：向量 + BM25，用 RRF 融合，再经 LLM 重排，并设有相关度门槛
- **引用溯源**：每个回答都附来源（文档名、页码范围、相关度评分）
- **流式输出**：SSE 逐字返回，体验同 ChatGPT
- **多轮对话**：对话记录保存在 SQLite；追问会先被改写成完整问题再去检索
- **账号体系**：注册 / 登录、JWT 鉴权，用户之间数据完全隔离
- **结构化摘要**：OpenAI Responses API + JSON Schema，输出固定格式的文档摘要
- **评估框架**：命中率 hit@k、MRR、忠实度、正确性、拒答准确率，并能按题型分析

---

## 两条核心流程

系统本质上只做两件事：**存文档**和**回答问题**。

### 流程 A：上传 PDF（把书放进图书馆）

```
用户上传 PDF（需登录）
  → ① 校验：扩展名、文件头、大小 ≤ 25MB                 routers/upload.py
  → ② 保存到 data/uploads/
  → ③ 逐页提取文字；文字少于 20 字符的页面走 OCR        services/pdf_loader.py
       （pypdf → PyPDF2 依次兜底；OCR 用 pypdfium2 渲染 + RapidOCR 识别）
  → ④ 跨页切块：整份文档拼接后切分，每块 1000 字符、    services/chunking_service.py
       重叠 200 字符，记录起止页 page ~ page_end
  → ⑤ 每块转成 1536 维向量                              services/embedding_service.py
  → ⑥ 存入该用户独立的 FAISS 索引并落盘                 services/vector_store.py
  → ⑦ SQLite 记录文档信息（归属用户、页数、块数、状态）  models.py
```

### 流程 B：提问（让教授答题）

```
用户输入问题
  → ① 读取本次对话的历史记录                            services/conversation_memory.py
  → ② 问题改写："那第三点呢？" → 完整的独立问题         rag_service._condense_question
  → ③ 混合检索                                          services/retriever.py
       ├─ 向量检索 top-12（余弦相似度 < 0.2 的丢弃）     services/vector_store.py
       ├─ BM25 关键词检索 top-12（jieba 中文分词）       services/keyword_search.py
       ├─ RRF 融合两路排名
       └─ LLM 重排：逐段打 0–10 分，< 4 分的丢弃，取前 4  services/reranker.py
  → ④ 拼接提示词：只能依据资料回答 + 资料 + 历史 + 问题  services/prompt_templates.py
  → ⑤ gpt-4o-mini 生成回答，经 SSE 流式推送到浏览器      routers/chat.py
  → ⑥ 问答连同引用来源一起存库，供后续追问使用
```

核心编排逻辑在 `services/rag_service.py`，检索管线在 `services/retriever.py`。

---

## 在基础版上做的优化

基础版是标准的"纯向量检索 RAG"。以下是发现的 7 个不足及对应改进：

| # | 基础版的问题 | 改进方案 | 关键文件 |
|---|---|---|---|
| 1 | **扫描版 PDF 读不出来**（没有 OCR） | 文字少于阈值的页面先渲染成图片，再用 RapidOCR 识别（ONNX，中英文，纯 pip 安装） | `pdf_loader.py` |
| 2 | **只有向量检索**：搜型号、人名、术语等精确词效果差 | 向量 + BM25 混合检索，RRF 融合，再加 LLM 重排（结构化输出打分） | `retriever.py`、`keyword_search.py`、`reranker.py` |
| 3 | **没有相关度门槛**：永远返回 4 段，哪怕全不相关 | 两道门槛：向量余弦相似度 ≥ 0.2，重排分数 ≥ 4/10；全部被过滤时明确告诉模型"没找到" | `retriever.py` |
| 4 | **按文档过滤是"先搜后筛"**：只取 16 个结果再过滤，目标文档可能一段都筛不出来 | 每个用户一个独立索引；过滤时搜索整个索引（对暴力检索的 Flat 索引来说没有额外开销） | `vector_store.py` |
| 5 | **按页切块**：跨页的句子会被切断 | 整份文档拼接后再切，用字符偏移量反推每块的起止页（引用显示为 p.3–4）；补充中文标点作为切分点 | `chunking_service.py` |
| 6 | **没有登录鉴权**：任何人都能调用接口、消耗 API 费用 | 注册 / 登录；PBKDF2 加盐哈希存密码；JWT 存于 HttpOnly + SameSite Cookie；按用户隔离数据；可设邀请码；生产环境使用默认密钥时拒绝启动 | `auth_service.py`、`routers/auth.py` |
| 7 | **没有效果评估** | 自建评估框架（参考 RAGAS 的指标设计）：hit@k、MRR、LLM 评委打分的忠实度 / 正确性 / 拒答准确率，按题型分析，并配有自动核对标注页码的脚本 | `evaluation/` |

**评估过程中额外发现并修复的问题：**

- **中文路径导致向量库无法存盘**：FAISS 的 C++ 文件读写在 Windows 上打不开含中文的路径。改为先序列化成字节，再由 Python 写文件（写临时文件后原子替换）。
- **BM25 在短文档上失效**：经典 Okapi IDF 对出现在一半以上段落里的词会算出 0 或负数，导致只有两三块的文档关键词检索完全失效。改用 Lucene / Elasticsearch 的 IDF 公式。
- **OpenAI 请求没有超时**：一次请求卡死了 84 分钟。现在所有调用都有超时；重排超时 20 秒，且超时后自动退回到不重排的排序。

---

## 效果评估

### 实验设置

| 项 | 内容 |
|---|---|
| 文档 | *Understanding Machine Learning*（Shalev-Shwartz & Ben-David），449 页 → 1080 个块 |
| 测试集 | **40 题**，6 种题型：术语 8、概念（换说法）8、跨章节 5、公式 5、中文问英文书 6、书中无答案（拒答）8 |
| 标注 | 每题包含标准答案、答案所在页码和原文证据；`check_dataset.py` 自动核对证据确实出现在标注页上 |
| 开发集 | 另有 11 题开发集用于调试，与测试集的知识点不重叠；测试集不用于调参 |
| 模型 | gpt-4o-mini（生成 / 重排 / 评委）、text-embedding-3-small；每题取 k = 4 段 |

### 总体结果

| 配置 | hit@4 | MRR | 忠实度 | 正确性 | 拒答准确率 | 检索耗时（中位数） |
|---|---|---|---|---|---|---|
| 纯向量（基线） | 0.812 | 0.703 | 0.963 | 0.784 | 1.00 | 0.16 s |
| 向量 + BM25 | 0.875 | 0.719 | 0.975 | 0.825 | 1.00 | 0.15 s |
| **向量 + BM25 + 重排** | **0.938** | **0.880** | **1.000** | **0.928** | 1.00 | 1.51 s |

相对基线：hit@4 **+12.5 个百分点**（相对 +15%），MRR **+25%**，正确性 **+18%**。

### 按题型（hit@4 / 正确性）

| 题型 | 纯向量 | 向量 + BM25 | + 重排 |
|---|---|---|---|
| 术语（8） | 0.62 / 0.69 | **1.00** / 0.88 | 1.00 / **1.00** |
| 概念（8） | 0.75 / 0.72 | 0.75 / 0.75 | **0.88 / 0.91** |
| 公式（5） | 0.80 / 0.80 | 0.80 / 0.80 | 0.80 / **0.96** |
| 中文（6） | 1.00 / 0.83 | 0.83 / 0.80 | 1.00 / 0.80 |
| 跨章节（5） | 1.00 / 0.96 | 1.00 / 0.92 | 1.00 / 0.96 |
| 拒答（8） | – / 1.00 | – / 1.00 | – / 1.00 |

### 结论

- **BM25 在术语类问题上收益最大**（命中率 0.62 → 1.00）：专有名词正是向量检索的弱项。
- **在换了说法的概念题上 BM25 没有帮助，重排才有帮助**（正确性 0.72 → 0.91）。
- **用中文问英文文档时，BM25 反而引入噪音**（命中率 1.00 → 0.83），重排把它纠正了回来。
- **检索质量提升后幻觉减少**：基线有两题，检索到的资料不足以回答（一题没检索到正确页面，另一题正确页面只排在第 4），模型就用自身知识补全了答案；加上重排后，相关段落被排到前面，忠实度达到 1.00。
- **代价**：重排多一次 LLM 调用，检索耗时从 0.16 s 增加到 1.5 s。

### 局限

- 40 题仍属小样本。命中率的提升来自 4 道题（纯向量未命中而重排命中），反方向为 0 道；方向一致，但单独看命中率还不具备统计显著性。MRR 和正确性的差距更大。
- 正确性由 LLM 评委打分，存在误判（例如把"最大化方差"和"最小化重建误差"判为不同，实际上二者等价），整体分数偏低；但各配置受到的影响相同，横向比较仍然有效。
- 表中的"纯向量"运行在改进后的切块策略之上，因此衡量的是检索方式本身的差异。

### 复现

```bash
# 1) 核对数据集标注（不调用 API）
venv\Scripts\python.exe evaluation\check_dataset.py --pdf 教材.pdf --dataset evaluation\dataset_ml_book_test.json
# 2) 运行评估（约 15 分钟，gpt-4o-mini 费用约 0.1 美元）
venv\Scripts\python.exe evaluation\run_eval.py --pdf 教材.pdf --dataset evaluation\dataset_ml_book_test.json
# 只评估检索（更省钱），或只跑部分配置
venv\Scripts\python.exe evaluation\run_eval.py --pdf 教材.pdf --retrieval-only --configs vector,hybrid
```

评估使用独立的临时索引，结束后自动删除，不影响应用数据。详细结果保存在 `evaluation/results/`。

---

## 技术栈

### 基础技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.12 |
| Web 后端 | FastAPI、Uvicorn、Jinja2；SSE 流式输出 |
| LLM 编排 | LangChain 0.3（Document Loader、TextSplitter、VectorStore、Retriever、PromptTemplate、LCEL） |
| 大模型 | OpenAI gpt-4o-mini、text-embedding-3-small；Responses API 结构化输出 |
| 向量检索 | FAISS |
| 数据存储 | SQLite + SQLAlchemy 2.0 |
| 配置 / 校验 | Pydantic v2、pydantic-settings、python-dotenv |
| PDF 解析 | pypdf、PyPDF2 |
| 前端 | 原生 HTML / CSS / JavaScript、marked.js、DOMPurify |
| 测试 | pytest、FastAPI TestClient |

### 新增技术栈

| 类别 | 技术 | 用途 |
|---|---|---|
| 稀疏检索 | **rank-bm25**（Lucene IDF） | BM25 关键词检索 |
| 中文分词 | **jieba** | 为 BM25 切分中文 |
| 结果融合 | **RRF（Reciprocal Rank Fusion）** | 合并向量与 BM25 两路排名 |
| 重排 | **LLM Reranker**（LangChain `with_structured_output` + JSON Schema） | 对候选段落逐一打 0–10 分 |
| OCR | **RapidOCR**（ONNX Runtime）、**pypdfium2** | 扫描页渲染成图片并识别文字 |
| 鉴权 | **PyJWT**、PBKDF2-HMAC-SHA256（hashlib）、HttpOnly / SameSite Cookie | 登录与数据隔离 |
| 评估 | **LLM-as-a-judge**，自实现 hit@k / MRR / Faithfulness / Correctness | 量化 RAG 效果 |
| 工程 | `asyncio.to_thread`、请求超时与降级、轻量数据库迁移、FAISS 字节序列化 | 稳定性与兼容性 |

---

## 项目结构

```
pdf-chat/
├── main.py                    应用入口：路由挂载、页面、登录跳转、启动检查
├── config.py                  全部配置项（从 .env 读取）
├── database.py / models.py    SQLite 连接、轻量迁移；User / Document / Conversation / ChatMessage
├── schemas.py                 API 请求和响应的数据结构
├── routers/                   【接口层】
│   ├── auth.py                注册 / 登录 / 登出 / 当前用户
│   ├── upload.py              上传与入库流水线
│   ├── documents.py           文档列表 / 删除 / 结构化摘要
│   ├── chat.py                问答（普通 + SSE 流式）、对话管理
│   └── health.py              健康检查
├── services/                  【业务层】
│   ├── pdf_loader.py          文字提取 + OCR
│   ├── chunking_service.py    跨页切块
│   ├── embedding_service.py   文字 → 向量
│   ├── vector_store.py        按用户隔离的 FAISS 索引
│   ├── keyword_search.py      BM25 + jieba
│   ├── reranker.py            LLM 重排
│   ├── retriever.py           混合检索管线（向量 → BM25 → RRF → 重排 → top-k）
│   ├── prompt_templates.py    回答 / 问题改写 / 重排 / 摘要的提示词
│   ├── conversation_memory.py 对话记忆
│   ├── auth_service.py        密码哈希、JWT、当前用户依赖
│   └── rag_service.py         RAG 编排
├── evaluation/
│   ├── run_eval.py            评估脚本（多配置对比、按题型统计）
│   ├── check_dataset.py       标注页码自动核对
│   ├── dataset_ml_book_test.json   40 题测试集
│   └── dataset.example.json        11 题开发集
├── templates/ static/         前端页面（含登录页）
├── tests/                     35 个测试（检索、鉴权、OCR、切块等，离线运行）
└── learn/rag_minimal.py       50 行 RAG 最小原型（学习用）
```

---

## 快速开始

需要 Python 3.12（依赖锁定的版本在 3.14 上还没有安装包）和一个 OpenAI API Key。

```bash
# 1. 创建虚拟环境并安装依赖
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 配置
copy .env.example .env        # 然后填入 OPENAI_API_KEY，并把 SECRET_KEY 改成随机字符串

# 3. 启动
venv\Scripts\python.exe -m uvicorn main:app --reload
```

打开 http://127.0.0.1:8000 → 注册账号 → 上传 PDF → 开始提问。API 文档在 `/docs`，
页面上的 **Authorize** 按钮可以填入登录返回的 token。

> macOS / Linux：把 `venv\Scripts\python.exe` 换成 `venv/bin/python`。

---

## 配置项

完整列表见 `.env.example`，常用的如下：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | 切块大小与重叠 |
| `RETRIEVER_K` | 4 | 最终交给模型的段落数 |
| `RETRIEVAL_CANDIDATES` | 12 | 每路检索的候选数 |
| `HYBRID_SEARCH_ENABLED` | true | 是否启用 BM25 |
| `MIN_VECTOR_SIMILARITY` | 0.2 | 向量相似度门槛 |
| `RERANK_ENABLED` / `RERANK_MIN_SCORE` | true / 4 | 是否重排、重排分数门槛 |
| `OCR_ENABLED` / `OCR_MIN_CHARS` | true / 20 | 是否 OCR、触发 OCR 的字数阈值 |
| `OPENAI_TIMEOUT_SECONDS` / `RERANK_TIMEOUT_SECONDS` | 60 / 20 | 请求超时 |
| `SECRET_KEY` | 占位值 | JWT 签名密钥，**部署前必须修改** |
| `REGISTRATION_INVITE_CODE` | 空 | 设置后，注册必须填写邀请码 |

---

## API

除 `/api/health` 和注册 / 登录外，所有接口都需要登录（Cookie 或 `Authorization: Bearer <token>`）。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/auth/register` · `/login` · `/logout` | 注册 / 登录 / 登出 |
| GET | `/api/auth/me` | 当前用户 |
| POST | `/api/upload` | 上传一个或多个 PDF |
| GET / DELETE | `/api/documents` · `/api/documents/{id}` | 文档列表、详情、删除 |
| POST | `/api/documents/{id}/summary` | 结构化摘要 |
| GET / POST | `/api/conversations` | 对话列表、新建对话 |
| GET / DELETE | `/api/conversations/{id}[/messages]` | 对话消息、删除对话 |
| POST | `/api/chat` · `/api/chat/stream` | 提问（普通 / SSE 流式） |
| GET | `/api/health` | 健康检查 |

---

## 测试

```bash
venv\Scripts\python.exe -m pytest
```

共 35 个测试，**不需要 API Key**：检索相关测试使用离线的哈希向量模型代替 OpenAI。覆盖内容：

- 混合检索、RRF 融合、相似度门槛、重排过滤与失败降级
- 按文档过滤时不会遗漏结果、中文 BM25、中文路径下的索引落盘
- 用户之间的数据隔离、JWT 与 Cookie 鉴权
- 扫描页 OCR、跨页切块的页码映射

---

## 已知局限与后续方向

- **重排的成本**：LLM 重排的延迟约 1.5 s。可以换成本地 cross-encoder（如 bge-reranker）降低延迟和费用。
- **学习型稀疏检索**：可以尝试 SPLADE 等模型替代或补充 BM25，并用现有评估框架对比效果。
- **评估的可信度**：扩大测试集规模；换用更强的评委模型；把标准答案改写为"必须包含的要点"清单。
- **公式提取**：PDF 中的数学公式提取后会变成乱码，影响公式类问题。
- **部署**：增加按用户的限流、调用费用统计；如需多实例部署，可迁移到托管向量数据库（pgvector、Qdrant 等）。

---

## License

MIT
