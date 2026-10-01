# INSTRUCTION.md：零基础安装与使用指南

这份指南假设你**从没用过 Python、VS Code、Git、终端、虚拟环境或 API**。按顺序一步步做，
就能在自己的电脑上把 PDF Chat 跑起来。

> 💡 不用担心弄坏什么：这里安装的东西都是安全、免费、可以卸载的。

想了解系统原理和效果数据，请看 [README.md](README.md)；想了解代码设计，请看
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 目录

1. [这个应用能做什么](#1-这个应用能做什么)
2. [安装 Python 3.12](#2-安装-python-312)
3. [安装 Git](#3-安装-git)
4. [安装 VS Code 和扩展](#4-安装-vs-code-和扩展)
5. [在 VS Code 中打开项目](#5-在-vs-code-中打开项目)
6. [创建并激活虚拟环境](#6-创建并激活虚拟环境)
7. [安装依赖](#7-安装依赖)
8. [创建 `.env` 配置文件](#8-创建-env-配置文件)
9. [获取 OpenAI API Key 并充值](#9-获取-openai-api-key-并充值)
10. [启动应用](#10-启动应用)
11. [注册与登录](#11-注册与登录)
12. [使用各项功能](#12-使用各项功能)
13. [运行自动测试](#13-运行自动测试)
14. [运行效果评估](#14-运行效果评估)
15. [故障排查](#15-故障排查)
16. [常见问题 FAQ](#16-常见问题-faq)
17. [常见错误](#17-常见错误)
18. [安全建议](#18-安全建议)
19. [下一步学什么](#19-下一步学什么)

---

## 1. 这个应用能做什么

**PDF Chat** 是一个网页应用，你可以：

- 注册账号并登录（每个人只能看到自己的文档和对话）；
- 上传 PDF，**扫描版 PDF 也可以**（会自动做文字识别 OCR）；
- 在聊天界面里针对 PDF 提问，AI 只根据文档内容回答，并标出**出自哪份文档的第几页**，以及这段资料的相关度评分。

应用运行在你自己的电脑上（localhost）。PDF、对话记录和向量索引都保存在本地，OCR 也在本地完成；
只有回答问题和生成向量时，才会把需要的文字发送给 OpenAI。

---

## 2. 安装 Python 3.12

> ⚠️ **请安装 3.12 版本，不要装最新版。**项目依赖的部分库（numpy、faiss、onnxruntime）
> 锁定的版本在更新的 Python（如 3.14）上还没有现成的安装包，`pip install` 会失败。
> 电脑上同时装着多个版本没关系，后面会指定使用 3.12。

### Windows

1. 打开 **https://www.python.org/downloads/**，找到 **Python 3.12.x**，下载 Windows installer (64-bit)。
2. 运行安装程序。**重要：**在第一个界面底部勾选 **"Add python.exe to PATH"**，再点 "Install Now"。
3. 安装完成后，按 `Win` 键，输入 `cmd` 打开命令提示符，输入：
   ```
   py -3.12 --version
   ```
   看到 `Python 3.12.x` 就说明安装成功。

### macOS

1. 在 **https://www.python.org/downloads/** 下载 Python 3.12.x 的 macOS 安装包（.pkg），按提示安装。
2. 打开**终端**（`Cmd + 空格`，输入 Terminal），输入：
   ```
   python3.12 --version
   ```

---

## 3. 安装 Git

Git 用来下载和管理代码。

- **Windows**：打开 **https://git-scm.com/download/win**，安装时一路点 "Next" 即可。
- **macOS**：在终端输入 `git --version`；如果没有安装，系统会提示你安装 "Command Line Developer Tools"，点击安装。

验证：输入 `git --version`，能看到版本号就说明装好了。

---

## 4. 安装 VS Code 和扩展

1. 打开 **https://code.visualstudio.com/** 下载并安装。
2. 打开 VS Code，点左侧栏的**扩展图标**（四个方块），搜索并安装：
   - **Python**（Microsoft）
   - **Pylance**（Microsoft，通常会随 Python 扩展一起安装）
   - 可选：**Jinja**，让 `templates/` 里的 HTML 模板有语法高亮

---

## 5. 在 VS Code 中打开项目

1. 如果项目还是 `.zip` 压缩包，先解压。
2. VS Code 菜单 **File → Open Folder…**，选择 `pdf-chat` 文件夹（里面有 `main.py`、`requirements.txt`）。
3. 菜单 **Terminal → New Terminal**，底部会打开一个终端。**后面所有命令都在这里输入。**

---

## 6. 创建并激活虚拟环境

**虚拟环境**是这个项目专用的 Python"沙盒"，避免它的依赖和电脑上的其他项目冲突。

**创建**（只需要做一次）：

```
# Windows
py -3.12 -m venv venv

# macOS
python3.12 -m venv venv
```

这会在项目里生成一个 `venv` 文件夹（已经在 `.gitignore` 里，不会被提交到 Git）。

**激活**（每次新开终端都要做）：

```
# Windows（命令提示符）
venv\Scripts\activate

# Windows（PowerShell）
venv\Scripts\Activate.ps1

# macOS
source venv/bin/activate
```

✅ 看到命令行开头出现 `(venv)`，就说明激活成功了。

> 如果 PowerShell 提示"禁止运行脚本"，先执行一次
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，再重新激活。
>
> **不想每次都激活？**本指南后面的命令都写成 `venv\Scripts\python.exe ...`
> （macOS 是 `venv/bin/python ...`），这样写会直接使用虚拟环境里的 Python，不激活也能用。

---

## 7. 安装依赖

```
venv\Scripts\python.exe -m pip install -r requirements.txt
```

大约需要几分钟，会下载 FastAPI、LangChain、FAISS、OpenAI SDK，以及 OCR 用到的
onnxruntime、opencv 等，其中 OCR 相关的库体积较大。屏幕上会滚过很多文字，这是正常的。

---

## 8. 创建 `.env` 配置文件

`.env` 是保存 API Key 等私密设置的文件。项目里只提供了模板 `.env.example`。

1. 复制模板：
   - Windows：`copy .env.example .env`
   - macOS：`cp .env.example .env`
2. 在 VS Code 里打开 `.env`，修改下面两项：
   - `OPENAI_API_KEY=`：填你的 OpenAI Key（下一节教你获取）；
   - `SECRET_KEY=`：用来给登录凭证签名，**改成一串随机字符**。可以用下面的命令生成一串：
     ```
     venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
     ```
3. 其他设置保持默认就行。每一项的含义见 README 的[配置项](README.md#配置项)。
   如果只想让熟人注册，可以给 `REGISTRATION_INVITE_CODE` 设一个邀请码。

> 修改 `.env` 后要**重启**应用，新设置才会生效。

---

## 9. 获取 OpenAI API Key 并充值

1. 登录 **https://platform.openai.com/api-keys**，点 **"Create new secret key"**。
2. **立刻复制**这个以 `sk-` 开头的 Key（它只显示一次），粘贴到 `.env`：
   ```
   OPENAI_API_KEY=sk-你的key
   ```
   注意不要加引号，等号两边也不要有空格。
3. **充值**：在 **Settings → Billing** 里充值。**账户余额为 0 时，上传和问答都会失败**
   （错误信息是 `429 ... You have no credits`）。学习用的话，充 5 美元就够用很久。
4. 建议在 Billing 里设置每月消费上限。

> 💰 参考费用：用 gpt-4o-mini，一次问答通常不到 0.1 美分；上传一本 450 页的书
> （生成向量）大约 0.5 美分；跑一次 40 题的完整评估大约 0.1 美元。

---

## 10. 启动应用

### 方式 A：双击启动脚本

- **Windows**：在文件资源管理器里双击 `Start App.bat`
- **macOS**：双击 `Start App (Mac).command`（第一次可能被系统拦截，见[故障排查](#15-故障排查)）

脚本会自动完成：检查 Python（优先使用 3.12）→ 创建虚拟环境 → 安装依赖 → 检查 `.env` → 启动应用 → 打开浏览器。

### 方式 B：手动启动（推荐，出错时能看到完整信息）

```
venv\Scripts\python.exe -m uvicorn main:app --reload
```

`--reload` 的作用是：修改代码后，服务器会自动重新加载。

### 打开应用

看到 `Uvicorn running on http://127.0.0.1:8000` 后，在浏览器里打开：

- **http://127.0.0.1:8000**：首页
- **http://127.0.0.1:8000/docs**：交互式 API 文档

停止应用：回到终端，按 `Ctrl + C`。

---

## 11. 注册与登录

1. 打开 http://127.0.0.1:8000/chat，没有登录时会自动跳转到登录页。
2. 切换到 **Create account**：
   - 用户名：3 个字符以上，只能用字母、数字和 `_ . -`；
   - 密码：8 个字符以上；
   - 如果设置了邀请码，还要填写邀请码。
3. 注册成功后会自动登录。登录状态默认保持 7 天。
4. 左下角显示你的用户名，旁边的 **Log out** 用来退出登录。

> 每个账号的文档、对话和检索索引是完全隔离的，看不到别人的数据。

---

## 12. 使用各项功能

### 上传 PDF

1. 点侧边栏的 **Upload**，把 PDF 拖进上传区域（也可以点击选择文件，可以一次选多个）。
2. 等状态从 "Processing…" 变成 **Ready**，下方会显示页数和切块数。
   - **扫描版 PDF** 会自动 OCR。第一次 OCR 需要加载模型，扫描页越多，处理越慢，请耐心等待。
   - 显示 **Failed** 时会写明原因，见[故障排查](#15-故障排查)。

### 提问

1. 打开 **Chat** 页面。
2. 在输入框上方的 **Scope** 下拉框里选择提问范围：**All documents**（所有文档）或某一份文档。
3. 输入问题后按 Enter 发送（Shift + Enter 换行），回答会逐字显示出来。
4. 点击回答下方的 **"📚 N sources"**，可以展开查看引用来源：
   - 文档名、页码（跨页时显示为 `p.12–13`）；
   - **relevance x/10**：重排模型给这段资料打的相关度分数；
   - 原文片段。
5. 可以接着追问，比如"那第二点呢？"，系统会结合上下文理解你的意思。
6. 如果文档里没有答案，AI 会直接说找不到，而不是编一个答案。

### 管理对话

- **+ New Chat**：开始新对话；
- 侧边栏的 Conversations 列出历史对话（保存在数据库里，重启后还在），点击即可继续聊。

### 管理文档

- 侧边栏的 Your Documents 显示每份文档的状态（绿 = 就绪，黄 = 处理中，红 = 失败）；
- 点击一份文档，可以把提问范围限定到这份文档；再点一次取消限定；
- 鼠标悬停时出现 **×**，点击可以删除文档（同时删除文件和检索索引）。

### 结构化摘要和 API（在 /docs 中使用）

结构化摘要功能**只提供了 API，网页上没有按钮**：

1. 先在网页上登录，然后打开 http://127.0.0.1:8000/docs，它会使用同一个登录状态；
2. 找到 `GET /api/documents`，点 **Try it out → Execute**，复制一份文档的 `id`；
3. 找到 `POST /api/documents/{document_id}/summary`，填入这个 id 并执行，
   就会返回包含标题、概述、要点、文档类型和预计阅读时间的 JSON 摘要。

如果用其他程序调用 API：先调用 `POST /api/auth/login` 拿到 `access_token`，之后的请求带上请求头
`Authorization: Bearer <token>`。在 `/docs` 页面上，也可以点右上角的 **Authorize** 把 token 填进去。

---

## 13. 运行自动测试

```
venv\Scripts\python.exe -m pytest
```

一共 35 个测试，全部显示 `PASSED` 就说明一切正常。测试**不需要 API Key，也不花钱**：
检索相关的测试用一个离线的"假向量模型"代替 OpenAI。测试使用临时数据库，不会碰你的真实数据。

---

## 14. 运行效果评估

评估脚本用一组标注好答案的问题，量化比较三种检索方式（纯向量 / 向量 + BM25 / 再加重排）的效果。
项目自带一个 40 题的测试集，针对教材 *Understanding Machine Learning*。

```
# 第 1 步：核对数据集的标注页码（不调用 API，不花钱）
venv\Scripts\python.exe evaluation\check_dataset.py --pdf 教材.pdf --dataset evaluation\dataset_ml_book_test.json

# 第 2 步：运行评估（约 15 分钟，费用约 0.1 美元）
venv\Scripts\python.exe evaluation\run_eval.py --pdf 教材.pdf --dataset evaluation\dataset_ml_book_test.json

# 省钱版：只评估检索，不生成回答、不打分
venv\Scripts\python.exe evaluation\run_eval.py --pdf 教材.pdf --dataset evaluation\dataset_ml_book_test.json --retrieval-only
```

跑完后终端会打印两张表（总成绩、按题型的成绩），完整记录保存在 `evaluation/results/`。
每个指标的含义见 README 的[效果评估](README.md#效果评估)。

**想用自己的 PDF 出题？**参照 `evaluation/dataset_ml_book_test.json` 的格式写一个 JSON 文件，
每题包括问题、标准答案、答案所在页码，以及一句原文证据（书里没有答案的题，标准答案写 `null`）。
写好后先用 `check_dataset.py` 核对页码。

---

## 15. 故障排查

**`pip install` 报错，提到 numpy、faiss 或 onnxruntime 装不上**
→ 多半是虚拟环境用了太新的 Python。删掉 `venv` 文件夹，用 `py -3.12 -m venv venv` 重新创建（见第 6 节）。

**`py -3.12` 提示 "No suitable Python runtime found"，但我确定装了 3.12**
→ 你的 3.12 可能是用 uv 等工具安装的，没有注册成 `-3.12` 这个名字。用 `py -0` 查看已安装的版本，
再按列出的名字指定，比如 `py -V:Astral/CPython3.12.13 -m venv venv`。
双击 `Start App.bat` 时会出现 `[WARN]` 提示，原因也是这个。

**上传或提问失败，提示 `429 ... You have no credits`**
→ OpenAI 账户余额为 0，去 Billing 页面充值（见第 9 节）。

**提示 `401 ... invalid_api_key` 或 `Your API key has been invalidated`**
→ Key 填错了、被删除了，或者被 OpenAI 自动吊销（比如 Key 被发布到了公开网站）。
新建一个 Key，替换 `.env` 里的旧 Key，然后重启应用。

**打开 /chat 一直跳转到登录页**
→ 还没有登录，或者登录已过期（默认 7 天）。重新登录即可。
修改过 `SECRET_KEY` 后，之前所有的登录状态都会失效，这是正常的。

**升级到新版后，以前上传的文档看不到了**
→ 新版加入了账号体系，旧文档没有归属用户，所以不会显示。登录后重新上传即可。

**上传后显示 Failed**
→ 看失败原因：
- 文件加了密码或已损坏 → 用 PDF 阅读器打开确认一下；
- "even with OCR" → 页面是空白的，或者扫描质量太差，识别不出文字；
- OpenAI 相关错误 → 见上面几条。

**`Address already in use`，或者 8000 端口被占用**
→ 已经有一个实例在运行（可能在另一个终端窗口里）。关掉那个实例，或者换个端口：
`venv\Scripts\python.exe -m uvicorn main:app --reload --port 8001`。

**回答很慢**
→ 每次提问会依次调用 OpenAI 做问题改写、向量检索、重排和生成，通常需要几秒。
如果更看重速度，可以在 `.env` 里设 `RERANK_ENABLED=false`，能省约 1.5 秒，但检索准确率会下降
（具体差多少见 README 的评估结果）。

**macOS 拦截 `Start App (Mac).command`（"来自身份不明的开发者"）**
→ 右键点击文件 → 打开 → 在弹窗里再点"打开"。只需要做一次。

**双击 `.bat` 后窗口一闪就关了**
→ 在命令提示符里运行它，就能看到报错：先 `cd` 到项目目录，再输入 `"Start App.bat"`。

---

## 16. 常见问题 FAQ

**需要花钱吗？**
软件全部免费。OpenAI 按用量收费，参考第 9 节的费用估算。

**我的数据安全吗？**
PDF、对话和索引都保存在本地的 `data/` 文件夹，OCR 也在本地运行。
只有生成向量和回答问题时，才会把相关的文字片段发送给 OpenAI。

**数据保存在哪里？**

| 内容 | 位置 |
|---|---|
| 上传的 PDF | `data/uploads/` |
| 检索索引（每个用户一个） | `data/vector_store/<用户id>/faiss_index.bin` |
| 用户、文档、对话记录 | `data/pdf_chat.db`（SQLite 数据库） |
| 评估结果 | `evaluation/results/` |

**怎么彻底重置（删除所有账号、文档和对话）？**
停止应用，删除 `data/uploads/` 和 `data/vector_store/` 里除 `.gitkeep` 以外的内容，再删除
`data/pdf_chat.db`。重新启动后，应用会自动重建空的数据库。

**能换成其他模型提供商吗？**
需要修改代码：`services/embedding_service.py`、`services/rag_service.py`、`services/reranker.py`。
注意：**更换向量模型后，已有的向量索引会失效**，需要重新上传文档。

**怎么停止应用？**
在运行它的终端里按 `Ctrl + C`。

---

## 17. 常见错误

- ❌ 用 Python 3.13 / 3.14 创建虚拟环境 → 依赖装不上，请用 3.12。
- ❌ 没激活虚拟环境就运行 `pip` 或 `uvicorn`（也没用 `venv\Scripts\python.exe`）→ 提示 "module not found"。
- ❌ 文件名不对：应该是 `.env`，不是 `env.example` 或 `.env.txt`。
- ❌ 在 `.env` 里给 Key 加引号，比如 `OPENAI_API_KEY="sk-..."` → 应该写成 `OPENAI_API_KEY=sk-...`。
- ❌ 把 `.env` 提交到 Git 或发到网上 → 泄露的 Key 很快会被 OpenAI 自动吊销，还可能被别人盗用。
- ❌ 修改 `.env` 后没有重启应用。
- ❌ 账户余额为 0 时提问。

---

## 18. 安全建议

- **绝不要分享 `.env` 和 API Key。**如果泄露了，立刻到 OpenAI 后台吊销，再新建一个。
- **一定要修改 `SECRET_KEY`。**它用来给登录凭证签名，用默认值的话，别人可以伪造任何用户的登录状态。
  设置 `APP_ENV=production` 时，如果 `SECRET_KEY` 还是默认值，应用会拒绝启动。
- **公开部署时：**
  - 开启 HTTPS（在 `production` 模式下，登录 Cookie 只会通过 HTTPS 发送）；
  - 设置 `REGISTRATION_INVITE_CODE`，避免陌生人注册、消耗你的 API 额度；
  - 把 `ALLOWED_ORIGINS` 改成具体的域名，不要用 `*`。
- 在 OpenAI 后台设置每月消费上限。

---

## 19. 下一步学什么

1. **动手跑最小原型**：`learn/rag_minimal.py` 用 50 行代码实现了 RAG 的核心 6 步：
   ```
   venv\Scripts\python.exe learn\rag_minimal.py 你的文件.pdf
   ```
   读懂它，再试着关掉文件，自己从头写一遍。
2. **按这个顺序读代码**：
   `services/rag_service.py` → `services/retriever.py` → `services/keyword_search.py` →
   `services/reranker.py` → `services/chunking_service.py` → `services/vector_store.py` →
   `routers/chat.py` → `main.py`
3. **读 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**：每个设计决定背后的原因。
4. **做实验，用评估脚本验证**：在 `.env` 里改 `CHUNK_SIZE`、`RETRIEVAL_CANDIDATES`、`RERANK_MIN_SCORE`，
   用开发集 `evaluation/dataset.example.json` 跑评估，看数字怎么变。
   注意：调参时不要用 40 题的测试集，否则测试集上的成绩就不可信了。
5. **系统学习**：
   - Python 官方教程：https://docs.python.org/3/tutorial/
   - FastAPI 教程：https://fastapi.tiangolo.com/
   - LangChain 文档：https://python.langchain.com/
   - Git 入门：https://docs.github.com/en/get-started

祝你玩得开心！🎉
