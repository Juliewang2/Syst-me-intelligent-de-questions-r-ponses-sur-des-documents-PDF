# 架构设计说明

本文是对根目录 [README.md](../README.md) 的补充，重点说明**为什么这样设计**。
安装和使用方法见 [INSTRUCTION.md](../INSTRUCTION.md)。

---

## 1. 分层结构

```
┌─────────────────────────────────────────────────────────────────────┐
│ 表现层   templates/*.html + static/js + static/css                    │
│          Jinja2 服务端渲染页面、原生 JS、SSE 流式客户端、登录页        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP / SSE（HttpOnly Cookie 或 Bearer Token）
┌───────────────────────────────▼─────────────────────────────────────┐
│ 接口层   routers/  auth · upload · documents · chat · health          │
│          main.py：页面路由（未登录跳转 /login）、启动检查              │
│          ── 鉴权依赖 get_current_user（services/auth_service.py）──   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ 业务层   services/                                                    │
│   入库：pdf_loader(+OCR) → chunking_service → embedding_service       │
│         → vector_store                                                │
│   检索：retriever = vector_store + keyword_search(BM25)               │
│                     → RRF → reranker                                  │
│   生成：rag_service（问题改写、提示词、流式输出）、prompt_templates   │
│   其他：conversation_memory、auth_service                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│ 持久层   SQLite：users · documents · conversations · chat_messages     │
│          FAISS：data/vector_store/<user_id>/faiss_index.bin（每用户）  │
│          文件：data/uploads/<document_id>.pdf                          │
└─────────────────────────────────────────────────────────────────────┘
```

依赖方向自上而下：业务层不依赖 FastAPI，所以评估脚本 `evaluation/run_eval.py`
可以直接调用 `retrieve()`、`answer_from_documents()`，不需要启动 Web 服务。

---

## 2. 数据模型

| 表 | 关键字段 | 说明 |
|---|---|---|
| `users` | id、username（唯一）、password_hash | 密码哈希格式为 `pbkdf2_sha256$迭代次数$盐$摘要` |
| `documents` | id、**owner_id**、original_filename、page_count、chunk_count、status、summary | status 取值为 processing / ready / failed |
| `conversations` | id、**owner_id**、document_id（可空）、title | document_id 为空表示在全部文档中检索 |
| `chat_messages` | id、conversation_id、role、content、sources（JSON） | 引用来源随消息一起保存，重新打开对话时仍能看到 |

每个 chunk 在向量索引中携带的 metadata：
`document_id`、`document_name`、`page`、`page_end`、`chunk_index`。
检索时还会在副本上附加 `vector_similarity` 和 `rerank_score`。

---

## 3. 请求流程：上传 PDF

1. `POST /api/upload`：鉴权后接收一个或多个 multipart 文件。
2. `routers/upload.py`：校验扩展名，分块写入磁盘并检查大小上限，再校验文件头 `%PDF-`。
3. 在 SQLite 中写入一条 `Document` 记录（`owner_id` 为当前用户，status = processing）。
4. `pdf_loader.extract_pdf_text`：
   - 依次尝试 LangChain `PyPDFLoader` → `pypdf` → `PyPDF2`，任一成功即可；
   - 文字少于 `OCR_MIN_CHARS` 的页面，用 `pypdfium2` 渲染成图片，再用 RapidOCR 识别。
5. `chunking_service.chunk_document`：跨页切块（见 [5.6](#56-跨页切块)）。
6. `vector_store.add_document_chunks(owner_id, chunks)`：生成向量，写入该用户的索引并落盘，
   同时更新索引版本号（BM25 缓存据此判断是否需要重建）。
7. status 更新为 ready；任何一步失败都写为 failed，并记录错误信息。

---

## 4. 请求流程：提问（流式）

1. `POST /api/chat/stream`，请求体为 `{conversation_id?, document_id?, message}`。
2. 校验 document_id 和 conversation_id 都属于当前用户，否则返回 404。
3. 读取最近 `MAX_HISTORY_MESSAGES` 条历史记录，并保存用户这条消息。
4. `rag_service.generate_answer_stream`：
   1. 在线程池中执行 `retrieve_context`（检索过程会发起阻塞的网络调用，不能卡住事件循环）；
   2. 有历史记录时，先用 `CONDENSE_QUESTION_PROMPT` 把追问改写成独立问题；
   3. `retriever.retrieve`，详见下一节；
   4. 把检索结果格式化成 `[Source i | 文件名 | page(s)]` 形式的上下文，填入 `RAG_PROMPT`，
      调用 `ChatOpenAI(streaming=True)`，用 `astream` 逐个 token 产出。
5. 路由把每一步包装成 SSE 事件：`start` → `sources` → `token`（多个）→ `close`，出错时发 `error`。
6. 流式结束后，用新的数据库会话保存回答和引用来源（请求级的会话此时可能已经关闭）。

**检索用改写后的问题，生成用原问题加历史。**检索需要一个完整、独立的问题；
而生成时模型能看到上下文，用户的原话反而更自然。

---

## 5. 检索管线（核心）

```
query ─┬─► 向量检索 top-12 ──► 过滤：余弦相似度 < 0.2 ─┐
       │   (FAISS, 用户索引, 可按文档过滤)             │
       │                                               ├─► RRF 融合 ─► 取前 12 ─► LLM 重排 ─► 过滤：分数 < 4 ─► top-4
       └─► BM25 检索 top-12 ──► 过滤：得分 = 0 ────────┘                        （超时或失败则保持融合顺序）
           (jieba 分词, Lucene IDF)
```

每一步都可以通过配置关闭，也可以按单次调用覆盖。评估脚本就是用 `hybrid` / `use_rerank`
这两个参数来对比三种配置的。整个管线同时被封装成 LangChain 的 `BaseRetriever`（`PDFChatRetriever`），
可以直接接入 LCEL 链。

### 5.1 为什么要混合检索

| | 向量检索（稠密） | BM25（稀疏） |
|---|---|---|
| 匹配依据 | 语义 | 字面词 |
| 擅长 | 换了说法的问题、跨语言问题 | 专有名词、编号、术语 |
| 不擅长 | 罕见术语、精确编号 | 同义改写、跨语言问题 |

评估结果印证了这张表：术语类问题，BM25 让命中率从 0.62 升到 1.00；
中文问英文书，BM25 反而让命中率从 1.00 降到 0.83。

### 5.2 为什么用 RRF 融合，而不是把分数加权相加

余弦相似度的范围是 0 到 1，BM25 的得分却没有上界，而且取值随文档集合变化，两种分数没法直接比较。
**RRF 只看排名**：`score = Σ 1/(60 + rank)`，所以不需要做分数归一化，也没有权重需要调。
常数 60 用来削弱"第 1 名和第 2 名"之间的差距，于是两路检索都排得靠前的段落，
会胜过只在一路里排第一的段落。

### 5.3 为什么还要重排，以及为什么用 LLM 重排

向量检索比较的是两个**预先算好**的向量，问题和段落其实从未"同时出现"在模型面前，所以速度快但比较粗糙。
重排模型把问题和每个候选段落**放在一起读**，判断更准，但速度太慢，没法用在整个索引上。
所以分成两步：先粗选 12 个候选，再精挑 4 个。

这里用 gpt-4o-mini 一次性给全部候选打分（`with_structured_output` + JSON Schema，strict 模式），
好处是不需要额外下载模型。代价是延迟增加约 1.4 秒。替代方案是本地 cross-encoder（如 bge-reranker）。

**重排是可选的增强。**它有单独的超时（20 秒）且不重试；调用失败时退回到 RRF 融合后的顺序，回答照常生成。

### 5.4 相关度门槛

基础版无论问题是否相关，都固定返回 4 段资料。现在有两道门槛：

- **向量余弦相似度 ≥ 0.2**。FAISS 的 Flat 索引返回的是平方 L2 距离，而 OpenAI 的向量是单位长度，
  所以可以换算：`cos = 1 − d²/2`。
- **重排分数 ≥ 4/10**。

如果所有段落都被过滤掉，上下文会明确写成"没有找到相关内容"，模型据此回答"文档中没有"。

### 5.5 检索纠错循环（`CORRECTIVE_RETRIEVAL_ENABLED`，默认关闭）

受 CRAG 启发：先判断检索结果好不好，不好就再检索一次。CRAG 训练了一个评估器、退回到网页搜索；
这里用已有的重排分数做评估，退回到"用改写后的查询在同一语料里再搜"——外交问题检索失败，
多半是说法（缩写、行话）或语言（法语问英文文件）的问题，而不是语料里没有答案。

| 第一轮最高重排分 | 判定 | 处理 |
|---|---|---|
| ≥ 7（upper） | CORRECT | 直接用（过滤后取 top-k） |
| lower ~ upper | AMBIGUOUS | 保留 ≥ lower 的段落，加上每条改写的检索结果，RRF 合并后对原问题重排一次 |
| < lower（默认 = `RERANK_MIN_SCORE`）或无候选 | INCORRECT | 同上，但不保留第一轮段落；最后仍无合格段落则返回空 |
| 全部无分（重排关闭或失败） | CORRECT | 没有依据可判断，保持原结果 |

- **改写**（`services/query_rewriter.py`）：先用 `normalize_un_symbol` 规范文号（不调用模型），
  再一次结构化调用生成"展开缩写的正式说法"和"语料中其他语言的翻译"，最多 3 条；失败返回空，不重试。
  改写看到的失败段落按密级选模型。
- **流式状态**：检索在工作线程里，`on_status` 用 `loop.call_soon_threadsafe` 把状态放进 `asyncio.Queue`，
  生成器同时等待"新状态"和"检索结束"，状态事件在 `sources` 之前发出；前端显示为灰色提示。
- **评估**：`hybrid+rerank+crag` 与 `hybrid+rerank+crag-noLang`（不翻译），新增 `trigger_rate`、`rescue_rate`；
  默认不跑这两个配置（会多花模型调用）。

### 5.6 跨页切块

先把各页文字用 `\n` 拼成全文（分页处往往不是段落结束，而是句子的中间），同时记录每页在全文中的起始偏移量。
`RecursiveCharacterTextSplitter` 开启 `add_start_index=True` 后，会返回每块在全文中的偏移量，
再用二分查找（`bisect`）就能把偏移量映射回 `page` 和 `page_end`。

另外两处细节：

- 切分点里补充了中文的 `。？！；`；
- 设置 `keep_separator="end"`，让句号跟着它所在的句子，而不是跑到下一块的开头。

### 5.7 BM25 的实现细节

- **分词**：`jieba.lcut_for_search` 能切分中文，英文单词保持原样；再过滤掉标点等不含字母数字的词。
- **IDF**：经典 Okapi 的 IDF 是 `log((N−n+0.5)/(n+0.5))`，当一个词出现在一半以上的段落里时，结果是 0 或负数。
  一个只有两三块的短文档，关键词检索会因此**完全失效**（这个问题是测试发现的）。
  所以改用 Lucene / Elasticsearch 的 `log(1 + (N−n+0.5)/(n+0.5))`，它永远是正数。
- **缓存**：BM25 索引从该用户的 FAISS docstore 构建，按用户缓存。
  向量索引每次修改都会让版本号加 1，BM25 发现版本号变了就重建。

---

## 6. 关键设计决策

### 6.1 每个用户一个 FAISS 索引（原来是全局共享一个）

基础版所有文档共用一个索引，按文档过滤时采用"先取 16 个结果，再按 metadata 过滤"
（LangChain 的 FAISS 是在近邻搜索**之后**才过滤）。如果其他文档的内容和问题更相关，
目标文档的段落可能一段都进不了这 16 个，过滤后就什么都不剩。

现在的做法：

- **按用户拆分索引**：用户之间物理隔离，不可能检索到别人的数据；每次搜索的范围也更小。
- **在用户索引内按文档过滤时，设 `fetch_k = ntotal`**，也就是搜索整个索引。
  对暴力检索的 Flat 索引来说，本来就要逐一计算所有向量的距离，所以没有额外开销。
- 测试 `test_document_filter_finds_chunks_buried_under_other_documents` 复现了原来的问题：
  60 段干扰内容 + 1 段目标内容。

如果数据规模大到需要近似最近邻索引（IVF / HNSW），应该换成支持"先过滤再搜索"的向量数据库，
比如 Qdrant、pgvector。

### 6.2 FAISS 的持久化方式

`FAISS.save_local` / `load_local` 由 C++ 层直接打开文件，在 Windows 上**无法处理含中文的路径**
（比如项目放在"大四实习"目录下）。现在改为 `serialize_to_bytes()`，再由 Python 写文件：
先写入 `.tmp` 临时文件，再用 `replace` 原子替换，这样即使中途崩溃，也不会留下写了一半的索引文件。

### 6.3 鉴权

| 选择 | 理由 |
|---|---|
| PBKDF2-HMAC-SHA256，39 万次迭代，每个用户随机盐 | 只用标准库 `hashlib`，没有额外依赖；用 `hmac.compare_digest` 做常量时间比较 |
| JWT（HS256，用 `SECRET_KEY` 签名，7 天过期） | 无状态，服务端不用保存会话 |
| 存在 HttpOnly + SameSite=Lax 的 Cookie 里 | JS 读不到，降低 XSS 窃取的风险；Lax 模式下跨站 POST 不会带上 Cookie，可以缓解 CSRF；生产环境加 `Secure` |
| 同时支持 `Authorization: Bearer` | 方便 API 客户端和 Swagger 的 Authorize 按钮 |
| 访问别人的资源时返回 404，而不是 403 | 不泄露"这个 id 是否存在" |
| 生产环境使用默认 `SECRET_KEY` 时拒绝启动 | 防止忘记修改密钥 |
| 可选邀请码 | 控制谁能注册，避免陌生人消耗 API 额度 |

### 6.4 超时与降级

所有 OpenAI 调用都设了超时（默认 60 秒，重试 2 次）。起因是评估中有一次重排请求卡了 84 分钟，
项目原本完全没有设置超时。重排设成 20 秒且不重试：它失败了也只是检索质量稍差一点，
不值得让用户一直等。

### 6.5 轻量数据库迁移

`create_all` 不会修改已经存在的表。`database._add_missing_columns()` 会对比模型和实际的表结构，
用 `ALTER TABLE ADD COLUMN` 补上缺少的列（比如 `owner_id`）。升级前已有的记录，`owner_id` 为空，
对所有用户都不可见。字段改动更复杂时，应该换成 Alembic。

### 6.6 结构化输出

有两处使用了 JSON Schema 约束输出：

- **文档摘要**：直接调用 OpenAI SDK 的 Responses API（`text.format.type = "json_schema"`，`strict = True`），
  对应 `POST /api/documents/{id}/summary`；
- **重排打分**：通过 LangChain 的 `with_structured_output(method="json_schema", strict=True)`。

两处都保证模型输出一定能解析成预期的结构，不需要写正则去"猜"内容。

### 6.7 部署模式与密级路由

所有聊天模型和 embedding 都由 `services/llm_factory.py` 创建，业务代码不再直接 new `ChatOpenAI`。

| `DEPLOYMENT_MODE` | 聊天模型 | embedding |
|---|---|---|
| openai（默认） | OpenAI | OpenAI |
| mistral | Mistral | Mistral |
| souverain | Ollama（本地） | Ollama（bge-m3） |
| hybride | 外部（默认 OpenAI），DR 上下文改走 Ollama | 始终本地 |

- **选模型的顺序**：模式默认 → 角色覆盖（`GENERATE_PROVIDER`、`RERANK_MODEL`…）→ 密级规则：
  上下文是 DR_SIMULE 且该 provider 不在 `DR_ALLOWED_PROVIDERS` 中，就改用 Ollama，调用记录里标 `rerouted`。
- **密级怎么算**（`services/sensitivity.py`）：任一片段是 DR_SIMULE 即为 DR；没有 classification 的片段也按 DR 处理，
  只有"openai 模式 + 未开启语料元数据"这一旧配置例外（此时什么都没有分级，行为保持不变）。
  对话历史也算：以前的回答若引用过 DR 来源，加载历史时会在 `AIMessage.response_metadata` 上打标记，
  下一轮的追问改写和生成就只交给本地模型。
- **入库**：embedding 会把每个片段发给 provider，所以 embedding provider 不允许接收 DR 时（openai、mistral 模式），
  DR 文件在上传、`ingest_corpus.py` 和 `index_document` 三处都会被拒绝。
- **结构化输出** `structured()`：OpenAI 用 json_schema 严格模式；Mistral 用 function_calling
  （langchain-mistralai 0.2.4 没有 json_schema，更新的版本要求 langchain-core 1.x）；
  Ollama 用 json_schema，temperature 0，并把 schema 附在 system 提示末尾。解析失败重试一次，仍失败抛 `StructuredOutputError`。
- **向量索引按 embedding 模型分目录**：`vector_store/<provider>__<model>/<user>/`，旁边的 `index_meta.json` 记录模型。
  模型不一致时报错，提示运行 `scripts/reindex.py`；旧布局的索引在 openai 模式下第一次访问时自动迁移。
- **外泄测试** `tests/test_leak.py`：hybride 模式下用记录输入的假外部模型跑完整流程（含流式、重排和带历史的追问），
  断言 DR 文本从未出现在外部模型的输入里，同时公开请求确实走了外部模型。
- 评委（`evaluation/run_eval.py`）不经过工厂，换部署模式不会改变评分标准。

---

## 7. 评估体系

```
dataset.json ──check_dataset.py──► 核对：证据句是否出现在标注页上（不调用 API）
     │
     ▼
run_eval.py
  ├─ 把 PDF 入库到临时索引（owner_id = eval_xxx，结束后删除）
  ├─ 每种配置 × 每道题：
  │    retrieve(hybrid=?, use_rerank=?) ──► hit@k、倒数排名（对照标注页码）
  │    answer_from_documents()          ──► LLM 评委 ──► 忠实度、正确性 / 拒答
  └─ 输出总表、按题型的表，以及 results/*.json（每题的检索页码、回答和评委理由）
```

- **页码比对**：段落的 `[page, page_end]` 和标注页有交集，就算命中。
- **开发集和测试集分开**：调参只用 11 题的开发集；40 题的测试集只用来报告成绩，避免"拿考题复习"。
- **延迟取中位数**：一次网络卡顿不会扭曲结果。
- **已知局限**：LLM 评委会误判（比如把等价的表述判为不同）；样本量仍然较小。详见 README。

---

## 8. 测试策略

- 35 个测试，**全部离线运行**。`tests/conftest.py` 提供了一个 `HashingEmbeddings`：
  把词哈希到 4096 个桶里、再归一化，作为假向量模型。共享词越多的文本，余弦相似度越高，
  足以测试检索逻辑，而不需要调用 OpenAI。
- 测试在启动前就设置好环境变量：临时数据库、临时索引目录、关闭重排，
  确保不会碰到 `data/` 里的真实数据。
- 需要重排的测试用 monkeypatch 替换 `rerank`，覆盖"过滤低分"和"失败降级"两种情况。
- 鉴权测试为每个用户创建独立的 `TestClient`（各自有自己的 Cookie），验证跨用户访问一律返回 404。

---

## 9. 扩展点

| 想要做的 | 修改位置 |
|---|---|
| 换向量数据库（Qdrant、pgvector…） | 重写 `services/vector_store.py`，接口保持不变 |
| 换成 Postgres | 修改 `DATABASE_URL`，并去掉 `database.py` 中 SQLite 专用的 `connect_args` |
| 本地 cross-encoder 重排 | 替换 `services/reranker.py` 中的 `rerank()` |
| 加入 SPLADE 等学习型稀疏检索 | 在 `retriever.retrieve` 中增加一路，加入 `ranked_lists`，由 RRF 统一融合 |
| 支持其他文件类型 | 在 `pdf_loader` 旁边增加对应的 loader，并让它输出相同的 `ExtractedPDF` 结构 |
| 用新配置做对比实验 | 在 `evaluation/run_eval.py` 的 `CONFIGS` 中增加一项 |
