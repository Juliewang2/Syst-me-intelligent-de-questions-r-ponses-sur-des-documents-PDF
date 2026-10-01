"""
services/prompt_templates.py
-------------------------------
Centralized LangChain `ChatPromptTemplate` definitions used across
the RAG pipeline. Keeping prompts here (rather than inline in the
service that uses them) makes them easy to test, version, and tune
independently of the retrieval/generation logic.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Main RAG answer-generation prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are "PDF Chat", a meticulous and friendly AI research assistant \
that answers questions strictly using the provided document excerpts.

Guidelines:
1. Answer ONLY using information found in the "Context" section below. Do not use outside \
knowledge, even if you believe you know the answer.
2. If the context does not contain enough information to answer confidently, say so plainly \
and suggest what the user could ask instead — do not guess or fabricate details.
3. When you use a fact from the context, keep your answer grounded and specific (quote figures, \
names, and dates exactly as they appear).
4. Format your answers using Markdown: use short paragraphs, bullet lists for enumerations, and \
**bold** for key terms when it improves readability.
5. Be concise but complete. Prefer a well-structured answer over a wall of text.
6. If the user's question is conversational (e.g. greetings, thanks) and not about the \
document, respond naturally and briefly without inventing document content.

Context from the uploaded document(s):
----------------------------------------
{context}
----------------------------------------
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Standalone-question rewriting prompt (used to make follow-up questions
# self-contained before they hit the retriever)
# ---------------------------------------------------------------------------

CONDENSE_QUESTION_SYSTEM_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up question to be a standalone question that captures all necessary context \
from the conversation. Do not answer the question — only rewrite it. If the follow-up question \
is already standalone, return it unchanged. Return ONLY the rewritten question, nothing else."""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_QUESTION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Follow-up question: {question}\n\nStandalone question:"),
    ]
)


# ---------------------------------------------------------------------------
# Reranking prompt (the chat model scores retrieved candidates 0-10)
# ---------------------------------------------------------------------------

RERANK_SYSTEM_PROMPT = """You are a search relevance judge. You will be given a user's question \
and numbered passages retrieved from their documents. Score EVERY passage for how useful it is \
for answering the question:

- 10: directly contains the answer
- 7-9: contains most of the answer or key supporting facts
- 4-6: related and partially useful
- 1-3: same general topic but does not help answer
- 0: unrelated

Judge only the passage text. Return a score for every passage number."""

RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RERANK_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nPassages:\n\n{passages}"),
    ]
)


# ---------------------------------------------------------------------------
# Structured-output document summary prompt (used with the OpenAI
# Responses API's structured output / JSON schema feature)
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are a precise document analyst. You will be given the text of a \
document (or an excerpt of it). Produce a structured summary strictly following the requested \
JSON schema. Be factual and avoid speculation. `key_points` should contain the most important, \
concrete takeaways from the document, ordered by importance."""


def build_summary_user_prompt(document_text: str, max_key_points: int) -> str:
    """Build the user-turn prompt for the structured summary generator."""
    return (
        f"Analyze the following document and produce a structured summary with at most "
        f"{max_key_points} key points.\n\n"
        f"DOCUMENT TEXT:\n\"\"\"\n{document_text}\n\"\"\""
    )
