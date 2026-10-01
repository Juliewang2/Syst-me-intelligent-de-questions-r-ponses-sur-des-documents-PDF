"""
RAG 进阶版学习原型 —— 新版系统的检索管线，压缩在一个文件里。

先读懂 rag_minimal.py（最基础的 6 步），再读这个文件。
和 minimal 相比，这里多了 5 个升级，每个都标着【升级 N】：

    【升级 1】OCR：扫描页没有文字层，先把页面渲染成图片再识别文字
    【升级 2】跨页切块：整本书拼起来再切，并记住每块从第几页到第几页
    【升级 3】BM25 关键词检索：弥补向量检索不擅长的专有名词、编号
    【升级 4】RRF 融合 + 相关度门槛：合并两路结果，丢掉不相关的
    【升级 5】LLM 重排：让大模型逐段打分，挑出最好的 4 段

为了好读，这里省略了正式项目里的工程细节（数据库、多用户、索引落盘、超时、流式输出等），
正式实现见 services/ 目录（每个升级都标了对应的文件）。

用法（在项目根目录）：
    # 问答模式：每个问题都会并排显示三种检索方式找到的页码，再用完整管线回答
    venv\\Scripts\\python.exe learn\\rag_advanced.py 你的文件.pdf

    # 迷你评估模式：用题集比较三种检索方式的 hit@4（只评检索，花费很少）
    venv\\Scripts\\python.exe learn\\rag_advanced.py 教材.pdf --eval evaluation\\dataset.example.json

需要 .env 里有真实的 OPENAI_API_KEY。每次运行都会重新给整份 PDF 生成向量
（一本 450 页的书大约 0.5 美分），因为这个学习版不把索引存盘。
"""

import json
import math
import re
import sys
from bisect import bisect_right

import jieba
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()
jieba.setLogLevel(60)  # 关掉 jieba 加载词典时的提示

CANDIDATES = 12        # 每路检索先粗选多少段
TOP_K = 4              # 最后交给大模型几段
MIN_SIMILARITY = 0.2   # 向量相似度门槛（余弦相似度）
MIN_RERANK_SCORE = 4   # 重排分数门槛（0-10 分）


# =====================================================================
# 第 1 步：读 PDF  +  【升级 1】OCR            正式版：services/pdf_loader.py
# =====================================================================
def read_pdf(path):
    """返回每页的文字列表。文字太少的页（多半是扫描图片）改用 OCR 识别。"""
    pages = []
    for page in PdfReader(path).pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # 个别页面可能让 pypdf 出错（正式版会换 PyPDF2 再试）
            pages.append("")

    scanned = [i for i, text in enumerate(pages) if len(text.strip()) < 20]
    if scanned:
        # 用到才导入：OCR 模型加载需要一两秒
        import pypdfium2
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        pdf = pypdfium2.PdfDocument(path)
        for i in scanned:
            image = pdf[i].render(scale=2).to_numpy()   # 页面 → 图片（放大 2 倍更清晰）
            result, _ = ocr(image)                      # 图片 → [[位置框, 文字, 置信度], ...]
            if result:
                pages[i] = "\n".join(item[1] for item in result)
        pdf.close()
        print(f"[1] 有 {len(scanned)} 页文字层为空，已用 OCR 识别")
    print(f"[1] 读取到 {len(pages)} 页")
    return pages


# =====================================================================
# 第 2 步：【升级 2】跨页切块                  正式版：services/chunking_service.py
# =====================================================================
def split_into_chunks(pages, source_name):
    """
    minimal 版是一页一页地切，跨页的句子会被切断。
    这里先把所有页拼成一整段文字，同时记下每页从第几个字符开始；
    切完后，再根据每块的起止字符位置，反查它属于第几页到第几页。
    """
    page_starts, texts, offset = [], [], 0
    for text in pages:
        page_starts.append(offset)
        texts.append(text)
        offset += len(text) + 1            # +1 是下面拼接时加的换行符
    full_text = "\n".join(texts)

    def page_of(char_index):               # 字符位置 → 页码（二分查找）
        return bisect_right(page_starts, char_index)   # 结果正好是从 1 开始的页码

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "。", "？", "！", " ", ""],  # 补上中文标点
        keep_separator="end",   # 句号跟着它所在的句子，不要跑到下一块的开头
        add_start_index=True,   # 关键：记录每块在全文中的起始位置
    )
    chunks = []
    for piece in splitter.create_documents([full_text]):
        start = piece.metadata["start_index"]
        end = start + len(piece.page_content) - 1
        chunks.append(Document(
            page_content=piece.page_content,
            metadata={"source": source_name, "id": len(chunks),
                      "page": page_of(start), "page_end": page_of(end)},
        ))
    print(f"[2] 切成 {len(chunks)} 块")
    return chunks


def pages_label(doc):
    p, q = doc.metadata["page"], doc.metadata["page_end"]
    return f"p.{p}" if p == q else f"p.{p}-{q}"


# =====================================================================
# 第 3 步：向量检索（带相似度门槛）            正式版：services/vector_store.py
# =====================================================================
def build_vector_index(chunks):
    index = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    print("[3] 向量索引建好了")
    return index


def vector_search(index, question):
    """返回 [(段落, 余弦相似度), ...]。"""
    results = index.similarity_search_with_score(question, k=CANDIDATES)
    # FAISS 返回的是"距离的平方"d，不是相似度。OpenAI 的向量长度都是 1，
    # 对这种向量有 d = 2 - 2*cos，所以 cos = 1 - d/2。
    scored = [(doc, 1 - float(d) / 2) for doc, d in results]
    # 【升级 4 的一半】相似度太低的直接丢掉：与其给模型不相关的资料，不如什么都不给
    return [(doc, s) for doc, s in scored if s >= MIN_SIMILARITY]


# =====================================================================
# 第 4 步：【升级 3】BM25 关键词检索           正式版：services/keyword_search.py
# =====================================================================
def tokenize(text):
    """jieba 能把中文切成词（中文没有空格）；英文单词会原样保留。去掉标点。"""
    return [w for w in jieba.lcut_for_search(text.lower()) if re.search(r"\w", w)]


class LuceneBM25(BM25Okapi):
    """
    BM25 的打分思路：问题里的词在这段里出现得越多，这段得分越高；
    越罕见的词（IDF 越大）越重要；段落越长，得分适当打折扣。

    坑：经典公式的 IDF = log((N-n+0.5)/(n+0.5))，当一个词出现在一半以上的段落里时，
    结果是 0 甚至负数。短文档只有两三块，关键词检索就会完全失效。
    所以改用 Elasticsearch 的版本 log(1 + ...)，它永远是正数。
    """
    def _calc_idf(self, nd):
        self.idf = {w: math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5)) for w, n in nd.items()}


def build_bm25_index(chunks):
    return LuceneBM25([tokenize(c.page_content) for c in chunks])


def keyword_search(bm25, chunks, question):
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:CANDIDATES]
    return [chunks[i] for i in best if scores[i] > 0]   # 一个词都没对上的不要


# =====================================================================
# 第 5 步：【升级 4】RRF 融合                   正式版：services/retriever.py
# =====================================================================
def rrf_fuse(*ranked_lists):
    """
    两路检索的分数没法直接比较（余弦相似度在 0~1 之间，BM25 得分没有上限），
    所以 RRF 只看排名：每段的得分 = 它在各路结果中 1/(60 + 名次) 的总和。
    两路都排得靠前的段落，会胜过只在一路里排第一的段落。
    """
    score, by_id = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata["id"]
            by_id[key] = doc
            score[key] = score.get(key, 0) + 1 / (60 + rank)
    return [by_id[k] for k in sorted(score, key=score.get, reverse=True)]


# =====================================================================
# 第 6 步：【升级 5】LLM 重排                   正式版：services/reranker.py
# =====================================================================
class PassageScore(BaseModel):
    index: int    # 第几段
    score: int    # 0-10 分


class RerankResult(BaseModel):
    scores: list[PassageScore]


RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是检索相关度评委。给每段资料打 0-10 分：10 = 直接包含答案，"
               "4-6 = 相关、部分有用，0 = 无关。每一段都必须打分。"),
    ("human", "问题：{question}\n\n{passages}"),
])


def rerank(question, docs):
    """
    向量检索比较的是两个预先算好的向量，问题和段落没有"同时出现"在模型面前，所以比较粗糙。
    重排让大模型把问题和每段放在一起读，判断更准，但速度慢，所以只用于粗选出来的候选。
    with_structured_output 强制模型按 RerankResult 的结构返回 JSON，程序可以直接用。
    """
    passages = "\n\n".join(f"[第 {i} 段]\n{d.page_content[:1200]}" for i, d in enumerate(docs, 1))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RERANK_PROMPT | llm.with_structured_output(RerankResult, method="json_schema", strict=True)
    try:
        result = chain.invoke({"question": question, "passages": passages})
    except Exception as e:  # 重排失败也不影响回答：退回融合后的顺序（正式版还设了超时）
        print("   (重排失败，使用融合顺序)", e)
        return [(d, None) for d in docs]
    score_of = {s.index: s.score for s in result.scores}
    scored = [(d, score_of.get(i, 0)) for i, d in enumerate(docs, 1)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    # 【升级 4 的另一半】分数太低的丢掉
    return [(d, s) for d, s in scored if s >= MIN_RERANK_SCORE]


# =====================================================================
# 把以上步骤组装成三种检索方式（就是评估里对比的三种配置）
# =====================================================================
def retrieve(question, index, bm25, chunks, mode):
    vector_hits = [doc for doc, _ in vector_search(index, question)]
    if mode == "vector":                                   # 基础版：只有向量
        return [(d, None) for d in vector_hits[:TOP_K]]
    fused = rrf_fuse(vector_hits, keyword_search(bm25, chunks, question))[:CANDIDATES]
    if mode == "hybrid":                                   # 向量 + BM25
        return [(d, None) for d in fused[:TOP_K]]
    return rerank(question, fused)[:TOP_K]                 # 向量 + BM25 + 重排


# =====================================================================
# 第 7 步：生成回答（带页码引用）                正式版：services/rag_service.py
# =====================================================================
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是文档助手。只能根据下面的资料回答，资料里没有就直说找不到，不要编造。"
               "引用时标注页码。\n\n资料：\n{context}"),
    ("human", "{question}"),
])


def answer(question, docs):
    if docs:
        context = "\n\n".join(f"[{pages_label(d)}] {d.page_content}" for d in docs)
    else:
        context = "（没有找到相关内容）"   # 明确告诉模型"没找到"，它才会老实说不知道
    chain = ANSWER_PROMPT | ChatOpenAI(model="gpt-4o-mini", temperature=0.2) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


MODES = ["vector", "hybrid", "hybrid+rerank"]


def chat_loop(index, bm25, chunks):
    while (question := input("\n你的问题（直接回车退出）：").strip()):
        print("\n三种检索方式找到的段落（按排名）：")
        results = {}
        for mode in MODES:
            results[mode] = retrieve(question, index, bm25, chunks, mode)
            shown = [pages_label(d) + (f"({s}分)" if s is not None else "") for d, s in results[mode]]
            print(f"  {mode:<14} {shown or '（全部低于门槛，什么都没找到）'}")
        final_docs = [d for d, _ in results["hybrid+rerank"]]
        print("\n回答（使用 hybrid+rerank 的结果）：\n" + answer(question, final_docs))


# =====================================================================
# 迷你评估：在题集上比较三种检索方式的 hit@4    正式版：evaluation/run_eval.py
# =====================================================================
def mini_eval(dataset_path, index, bm25, chunks):
    """
    hit@4：标准答案所在的页，有没有出现在检索到的 4 段里。
    这里只评"找资料"这一步，不评回答质量（那需要 LLM 评委，见正式版）。
    """
    questions = [q for q in json.load(open(dataset_path, encoding="utf-8")) if q.get("pages")]
    print(f"\n用 {len(questions)} 道有答案的题评估检索（每题取 {TOP_K} 段）：")
    for mode in MODES:
        hits = 0
        for q in questions:
            docs = [d for d, _ in retrieve(q["question"], index, bm25, chunks, mode)]
            hit = any(d.metadata["page"] <= p <= d.metadata["page_end"] for d in docs for p in q["pages"])
            hits += hit
        print(f"  {mode:<14} hit@{TOP_K} = {hits}/{len(questions)} = {hits / len(questions):.2f}")


if __name__ == "__main__":
    pdf_path = sys.argv[1]
    pages = read_pdf(pdf_path)
    chunks = split_into_chunks(pages, source_name=pdf_path)
    index = build_vector_index(chunks)
    bm25 = build_bm25_index(chunks)
    if len(sys.argv) >= 4 and sys.argv[2] == "--eval":
        mini_eval(sys.argv[3], index, bm25, chunks)
    else:
        chat_loop(index, bm25, chunks)
