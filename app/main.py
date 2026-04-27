from fastapi import FastAPI
from app.api.routes.chat import router as chat_router

app = FastAPI(title="AI Research Assistant", version="0.1.0")

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])

@app.get("/health")
def health():
    return {"status": "ok"}