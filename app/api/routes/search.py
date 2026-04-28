from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.hybrid_retriever import hybrid_search

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    k: int = 5

@router.post("/search")
def search(request: SearchRequest):
    try:
        results = hybrid_search(request.query, k=request.k)

        return {
            "query": request.query,
            "results": [
                {
                    "chunk_id": r["chunk_id"],
                    "source": r["source"],
                    "page": r["page"],
                    "text": r["text"][:200] + "...",  # preview only
                    "hybrid_score": round(r["hybrid_score"], 4),
                    "semantic_score": round(r["semantic_score"], 4),
                    "bm25_score": round(r["bm25_score"], 4),
                }
                for r in results
            ]
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))