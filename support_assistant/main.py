"""Offline-first Zepto policy assistant with LangGraph routing and FastAPI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"
POLICY_KEYWORDS = (
    "delivery", "return", "refund", "membership", "tracking", "cancel",
    "gift card", "support hours",
)

PROMPT_TEMPLATE = """Role: You are Zepto's policy support assistant.
Context: Use only the policy excerpts supplied below.
Task: Answer the user's question accurately and cite the supplied source IDs.
Format: Return JSON with answer (string), sources (list of IDs), and confidence (0-1).
Length: Keep the answer concise, at most 3 sentences.
Negative constraint: Do not answer using information not present in the provided context.
Few-shot example:
User: How long do refunds take?
Context: Approved refunds reach the original payment method within 3-5 business days.
Answer: {{"answer":"Approved refunds reach the original payment method within 3-5 business days.","sources":["doc_02"],"confidence":1.0}}

User question: {query}
Policy context: {context}
"""


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list[dict[str, str]]
    response: dict[str, Any]


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


class PolicyIndex:
    """Loads the eight documents and keeps embeddings in a persistent Chroma collection."""

    def __init__(self) -> None:
        import chromadb

        self.documents = self._read_documents()
        if len(self.documents) != 8:
            raise RuntimeError(
                f"Expected the 8 policy documents, found {len(self.documents)} in {DOCS_DIR}"
            )
        
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        if not self._collection_matches_documents():
            self._build()

    def _read_documents(self) -> list[dict[str, str]]:
        return [
            {"id": path.stem, "text": path.read_text(encoding="utf-8").strip()}
            for path in sorted(DOCS_DIR.glob("doc_*.txt"))
        ]

    def _build(self) -> None:
        ids = [document["id"] for document in self.documents]
        documents = [document["text"] for document in self.documents]
        metadatas = [{"document_id": document_id} for document_id in ids]
        embeddings = self.model.encode(documents, normalize_embeddings=True).tolist()
        existing_ids = self.collection.get(include=[])["ids"]
        stale_ids = [item_id for item_id in existing_ids if item_id not in ids]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        if ids:
            self.collection.upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

    def _collection_matches_documents(self) -> bool:
        expected = {document["id"]: document["text"] for document in self.documents}
        stored = self.collection.get(include=["documents"])
        actual = dict(zip(stored["ids"], stored["documents"] or []))
        return actual == expected

    def search(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(limit, self.collection.count()),
        )
        documents = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return [{"id": item_id, "text": text} for item_id, text in zip(ids, documents)]


index: PolicyIndex | None = None


def get_index() -> PolicyIndex:
    global index
    if index is None:
        index = PolicyIndex()
    return index


def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


def real_llm(prompt: str) -> str:
    """Call an OpenAI-compatible endpoint and return its raw text."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message.content or "{}"


def real_response(prompt: str) -> dict[str, Any]:
    """Validate the final schema and retry malformed LLM JSON twice."""
    for attempt in range(3):
        raw = real_llm(prompt)
        try:
            return AskResponse.model_validate_json(raw).model_dump()
        except Exception:
            prompt = (
                prompt + "\nYour previous response was invalid. Return only valid JSON "
                "with answer, sources, and confidence fields."
            )
            if attempt == 2:
                return AskResponse(
                    answer="ERROR: the optional LLM returned invalid structured output.",
                    sources=[], confidence=0.0,
                ).model_dump()
    raise RuntimeError("unreachable")


def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    if mock_mode():
        is_policy_query = any(keyword in query.lower() for keyword in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy_query else "general_question"
    else:
        raw = real_llm(
            "Classify this query as exactly policy_question or general_question. "
            f"Return JSON with an intent field. Query: {query}"
        )
        try:
            candidate = json.loads(raw).get("intent")
            intent = (
                candidate
                if candidate in {"policy_question", "general_question"}
                else "general_question"
            )
        except json.JSONDecodeError:
            intent = "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    retrieved = get_index().search(state["query"], limit=3)
    if mock_mode():
        if not retrieved:
            response = AskResponse(
                answer="I could not find a matching Zepto policy.",
                sources=[],
                confidence=0.0,
            )
            return {"retrieved": [], "response": response.model_dump()}
        answer = f"Based on the retrieved context: {retrieved[0]['text'][:200]}"
        response = AskResponse(
            answer=answer,
            sources=[item["id"] for item in retrieved],
            confidence=1.0,
        )
        return {"retrieved": retrieved, "response": response.model_dump()}
    context = "\n".join(f"[{item['id']}] {item['text']}" for item in retrieved)
    response = real_response(PROMPT_TEMPLATE.format(query=state["query"], context=context))
    return {"retrieved": retrieved, "response": response}


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_mode():
        response = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[], confidence=1.0,
        )
    else:
        response = AskResponse.model_validate(real_response(PROMPT_TEMPLATE.format(
            query=state["query"], context="No retrieval context is available."
        )))
    return {"response": response.model_dump()}


def route(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph() -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route)
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


graph = build_graph()
app = FastAPI(title="Zepto Policy Support Assistant")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = graph.invoke({"query": request.query})
    return AskResponse.model_validate(result["response"])


if __name__ == "__main__":
    print(json.dumps(app.openapi(), indent=2))
