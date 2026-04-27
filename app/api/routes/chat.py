from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest
from app.services.claude_service import chat_stream
from app.utils.prompt_builder import build_system_prompt

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = request.system_prompt or build_system_prompt()

    def stream_generator():
        for chunk in chat_stream(request.messages, system_prompt):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")