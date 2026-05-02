from fastapi import FastAPI
from app.api.routes.chat import router as chat_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.search import router as search_router
from app.api.routes.rag import router as rag_router
from app.api.routes.agent import router as agent_router
from app.api.routes.research import router as research_router
from app.api.routes.vision import router as vision_router

app = FastAPI(title="AI Research Assistant", version="0.1.0")

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(ingest_router, prefix="/api/v1", tags=["ingest"])
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(rag_router, prefix="/api/v1", tags=["rag"])
app.include_router(agent_router, prefix="/api/v1", tags=["agent"])
app.include_router(research_router, prefix="/api/v1", tags=["research"])
app.include_router(vision_router, prefix="/api/v1", tags=["vision"])

@app.get("/health")
def health():
    return {"status": "ok"}