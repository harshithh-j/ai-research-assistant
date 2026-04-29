from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.research_service import run_research

router = APIRouter()

class ResearchRequest(BaseModel):
    question: str
    k: int = 5

@router.post("/research")
def research(request: ResearchRequest):
    try:
        return StreamingResponse(
            run_research(request.question, k=request.k),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))