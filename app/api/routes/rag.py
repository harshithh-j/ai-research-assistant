from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.rag.hybrid_retriever import hybrid_search
from app.utils.prompt_builder import build_rag_prompt
from app.services.claude_service import rag_stream
from app.services.query_rewriter import rewrite_query, rewrite_with_expansion

router = APIRouter()

class RAGRequest(BaseModel):
    question: str
    k: int = 5
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    rewrite_query: bool = True
    expand_queries: bool = False

@router.post("/ask")
def ask(request: RAGRequest):
    try:
        original_question = request.question

        # Step 1: Query rewriting
        if request.expand_queries:
            # Generate 3 query variants, search with each, merge results
            queries = rewrite_with_expansion(original_question)
            print(f"Expanded queries: {queries}")

            seen_ids = set()
            chunks = []

            for q in queries:
                results = hybrid_search(
                    query=q,
                    k=request.k,
                    semantic_weight=request.semantic_weight,
                    bm25_weight=request.bm25_weight
                )
                for r in results:
                    if r["chunk_id"] not in seen_ids:
                        seen_ids.add(r["chunk_id"])
                        chunks.append(r)

            # Re-sort merged results by hybrid score and take top-k
            chunks = sorted(chunks, key=lambda x: x["hybrid_score"], reverse=True)[:request.k]
            search_query = queries[0]

        elif request.rewrite_query:
            # Single rewrite
            search_query = rewrite_query(original_question)
            print(f"Original: {original_question}")
            print(f"Rewritten: {search_query}")

            chunks = hybrid_search(
                query=search_query,
                k=request.k,
                semantic_weight=request.semantic_weight,
                bm25_weight=request.bm25_weight
            )

        else:
            # No rewriting — use original question directly
            search_query = original_question
            chunks = hybrid_search(
                query=search_query,
                k=request.k,
                semantic_weight=request.semantic_weight,
                bm25_weight=request.bm25_weight
            )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found."
            )

        # Step 2: Build prompt with context
        system_prompt, _ = build_rag_prompt(original_question, chunks)

        # Step 3: Build source list
        sources = "\n\nSources:\n" + "\n".join(
            f"[{i}] {c['source']} (page {c['page']})"
            for i, c in enumerate(chunks, 1)
        )

        # Step 4: Stream answer + sources
        def stream_with_sources():
            yield f"Search query used: {search_query}\n\n"
            for chunk_text in rag_stream(original_question, system_prompt):
                yield chunk_text
            yield sources

        return StreamingResponse(stream_with_sources(), media_type="text/plain")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rewrite")
def rewrite(payload: dict):
    """
    Debug endpoint — see how Claude rewrites a query.
    """
    question = payload.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    single = rewrite_query(question)
    expanded = rewrite_with_expansion(question)

    return {
        "original": question,
        "rewritten": single,
        "expanded": expanded
    }