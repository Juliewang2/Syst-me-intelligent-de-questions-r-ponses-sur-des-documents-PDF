"""
RAG 最小原型 —— 不要网页、不要数据库，只保留 RAG 的核心 6 步。
读懂这个文件，就读懂了整个项目的"灵魂"。

运行方式（在项目根目录）：
    venv\Scripts\python.exe learn\rag_minimal.py 你的文件.pdf

需要 .env 里有真实的 OPENAI_API_KEY。
"""

import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # 从 .env 读取 OPENAI_API_KEY

# ---------- 第 1 步：读 PDF ----------
# 每一页变成一个 Document 对象：page_content = 这页的文字，metadata = {"page": 页码, ...}
pages = PyPDFLoader(sys.argv[1]).load()
print(f"[1] 读取到 {len(pages)} 页")

# ---------- 第 2 步：切块 ----------
# 每块最多 1000 个字符，相邻两块重叠 200 个字符，防止一句话被从中间切断后丢失上下文
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"[2] 切成 {len(chunks)} 块，第一块开头：{chunks[0].page_content[:60]!r}")

# ---------- 第 3 步：向量化 + 存入 FAISS ----------
# 每块文字 -> 1536 个数字（向量）。意思相近的文字，向量也相近。
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("[3] 向量库建好了")

# ---------- 第 4 步：检索器 ----------
# 给一个问题，返回与它向量最接近的 k=4 块
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- 第 5 步：提示词模板 ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是文档助手。只能根据下面的资料回答，资料里没有就说不知道，不要编造。\n\n资料：\n{context}"),
    ("human", "{question}"),
])

# ---------- 第 6 步：大模型 + 串成管道（LCEL） ----------
# temperature 越低回答越稳定，RAG 场景要"老实"，所以设低
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
chain = prompt | llm | StrOutputParser()   # 填模板 -> 调模型 -> 取出纯文本

# ---------- 循环问答 ----------
while (question := input("\n你的问题（直接回车退出）：").strip()):
    docs = retriever.invoke(question)                       # 检索
    context = "\n\n".join(f"[第{d.metadata.get('page', 0) + 1}页] {d.page_content}" for d in docs)
    answer = chain.invoke({"context": context, "question": question})  # 增强 + 生成
    print("\n回答：", answer)
    print("来源页：", sorted({d.metadata.get("page", 0) + 1 for d in docs}))
