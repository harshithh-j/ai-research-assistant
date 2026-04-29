from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.tool_executor import run_agent

router = APIRouter()

class AgentRequest(BaseModel):
    question: str

@router.post("/agent")
def agent(request: AgentRequest):
    try:
        return StreamingResponse(
            run_agent(request.question),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))