from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.rag.hybrid_retriever import hybrid_search
from app.utils.prompt_builder import build_rag_prompt, format_chunks_as_context
from app.services.claude_service import rag_stream

router = APIRouter()

class RAGRequest(BaseModel):
    question: str
    k: int = 5
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

@router.post("/ask")
def ask(request: RAGRequest):
    try:
        # Step 1: Retrieve relevant chunks
        chunks = hybrid_search(
            query=request.question,
            k=request.k,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found. Try ingesting documents first."
            )

        # Step 2: Build prompt with context
        system_prompt, _ = build_rag_prompt(request.question, chunks)

        # Step 3: Build source list to append after streaming
        sources = "\n\nSources:\n" + "\n".join(
            f"[{i}] {c['source']} (page {c['page']})"
            for i, c in enumerate(chunks, 1)
        )

        # Step 4: Stream answer + append sources at end
        def stream_with_sources():
            for chunk_text in rag_stream(request.question, system_prompt):
                yield chunk_text
            yield sources

        return StreamingResponse(stream_with_sources(), media_type="text/plain")

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))