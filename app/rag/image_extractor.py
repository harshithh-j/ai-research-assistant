import fitz  # PyMuPDF
import base64
import io
from pathlib import Path
from typing import List, Dict
from PIL import Image

def extract_images_from_pdf(file_path: str, max_images: int = 10) -> List[Dict]:
    """
    Extracts images from a PDF file.
    Returns list of dicts with base64 encoded image data and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    images = []

    for page_num in range(len(doc)):
        if len(images) >= max_images:
            break

        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            if len(images) >= max_images:
                break

            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert to PNG using Pillow for consistency
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Resize if too large — Claude has image size limits
                max_size = (1024, 1024)
                pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                buffer.seek(0)
                b64_data = base64.standard_b64encode(buffer.read()).decode("utf-8")

                images.append({
                    "page": page_num + 1,
                    "index": img_index,
                    "source": path.name,
                    "media_type": "image/png",
                    "data": b64_data,
                    "width": pil_image.width,
                    "height": pil_image.height,
                })

            except Exception as e:
                print(f"Failed to extract image {img_index} from page {page_num}: {e}")
                continue

    doc.close()
    return images


def image_file_to_base64(file_path: str) -> Dict:
    """
    Converts an uploaded image file to base64 for Claude.
    Supports: PNG, JPEG, GIF, WEBP
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    ext = path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }

    media_type = media_type_map.get(ext, "image/png")

    # Open, resize if needed, convert to base64
    pil_image = Image.open(file_path)
    max_size = (1024, 1024)
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    fmt = "PNG" if ext == ".png" else "JPEG"
    pil_image.save(buffer, format=fmt)
    buffer.seek(0)

    b64_data = base64.standard_b64encode(buffer.read()).decode("utf-8")

    return {
        "source": path.name,
        "media_type": media_type,
        "data": b64_data,
        "width": pil_image.width,
        "height": pil_image.height,
    }