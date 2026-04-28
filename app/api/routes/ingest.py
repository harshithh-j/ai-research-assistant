from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.pdf_loader import load_pdf
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import save_index

router = APIRouter()

class IngestRequest(BaseModel):
    filename: str  # e.g. "research_paper.pdf"

@router.post("/ingest")
def ingest(request: IngestRequest):
    file_path = f"data/pdfs/{request.filename}"

    try:
        # Step 1: Load PDF
        pages = load_pdf(file_path)
        print(f"Loaded {len(pages)} pages")

        # Step 2: Chunk text
        chunks = chunk_text(pages)
        print(f"Created {len(chunks)} chunks")

        # Step 3: Embed chunks
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        print(f"Generated embeddings: {embeddings.shape}")

        # Step 4: Save to FAISS
        save_index(embeddings, chunks)

        return {
            "status": "success",
            "pages": len(pages),
            "chunks": len(chunks),
            "embedding_dim": embeddings.shape[1]
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))