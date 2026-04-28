import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    """
    Loads a PDF and extracts text page by page.
    Returns a list of dicts with page number and text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:  # skip empty pages
            pages.append({
                "page": page_num + 1,  # 1-indexed for human readability
                "text": text,
                "source": path.name
            })

    doc.close()
    return pages