import uuid
from pathlib import Path

from .base import Document


def load_text_file(path: str | Path) -> Document:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return Document(
        id=str(uuid.uuid4()),
        text=text,
        metadata={"source": str(path), "filename": path.name, "type": "text"},
    )


def load_pdf_file(path: str | Path) -> Document:
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(
        id=str(uuid.uuid4()),
        text=text,
        metadata={"source": str(path), "filename": path.name, "type": "pdf", "pages": len(reader.pages)},
    )


def load_directory(dir_path: str | Path, extensions: list[str] | None = None) -> list[Document]:
    if extensions is None:
        extensions = [".txt", ".md", ".pdf"]
    dir_path = Path(dir_path)
    docs = []
    for ext in extensions:
        for file in sorted(dir_path.glob(f"*{ext}")):
            if ext == ".pdf":
                docs.append(load_pdf_file(file))
            else:
                docs.append(load_text_file(file))
    return docs
