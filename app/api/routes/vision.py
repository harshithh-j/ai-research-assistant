from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, List
import tempfile
import os
from app.rag.image_extractor import image_file_to_base64, extract_images_from_pdf
from app.services.vision_service import run_vision

router = APIRouter()

@router.post("/vision/analyze")
async def analyze_image(
    question: str = Form(...),
    use_rag: bool = Form(True),
    k: int = Form(3),
    files: List[UploadFile] = File(...)
):
    """
    Analyze uploaded images with Claude vision.
    Accepts: PNG, JPEG, GIF, WEBP image files.
    """
    images = []

    for file in files:
        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Use PNG, JPEG, GIF, or WEBP."
            )

        # Save to temp file and convert to base64
        try:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            image_data = image_file_to_base64(tmp_path)
            image_data["source"] = file.filename
            images.append(image_data)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)  # clean up temp file

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    return StreamingResponse(
        run_vision(question, images, use_rag=use_rag, k=k),
        media_type="text/plain"
    )


@router.post("/vision/pdf-images")
async def analyze_pdf_images(
    question: str = Form(...),
    filename: str = Form(...),
    max_images: int = Form(5),
    use_rag: bool = Form(True),
    k: int = Form(3)
):
    """
    Extract images from an ingested PDF and analyze them.
    Useful for reading charts, diagrams, figures from research papers.
    """
    file_path = f"data/pdfs/{filename}"

    try:
        images = extract_images_from_pdf(file_path, max_images=max_images)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not images:
        raise HTTPException(
            status_code=404,
            detail="No images found in this PDF."
        )

    return StreamingResponse(
        run_vision(question, images, use_rag=use_rag, k=k),
        media_type="text/plain"
    )