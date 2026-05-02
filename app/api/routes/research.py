from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.research_service import run_research

router = APIRouter()

class ResearchRequest(BaseModel):
    question: str
    k: int = 5
    use_reranking: bool = True
    use_compression: bool = False  # off by default — costs extra API calls

@router.post("/research")
def research(request: ResearchRequest):
    try:
        return StreamingResponse(
            run_research(
                question=request.question,
                k=request.k,
                use_reranking=request.use_reranking,
                use_compression=request.use_compression,
            ),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))