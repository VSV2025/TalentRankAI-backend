"""Resume text extraction and structured field parsing."""
import hashlib
import re
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text(path: str) -> str:
    """Extract plain text from PDF or DOCX file."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def _extract_pdf(path: str) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not installed — returning empty text for PDF")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def _extract_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        logger.warning("python-docx not installed — returning empty text for DOCX")
        return ""
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""


def file_hash(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


SKILL_KEYWORDS = [
    "python", "pytorch", "tensorflow", "jax", "numpy", "pandas", "scikit-learn",
    "llm", "gpt", "bert", "transformers", "rlhf", "fine-tuning", "finetuning",
    "langchain", "langraph", "langgraph", "mlops", "mlflow", "kubeflow", "airflow",
    "kubernetes", "docker", "aws", "gcp", "azure", "sagemaker", "vertex ai",
    "distributed training", "cuda", "gpu", "c++", "rust", "go", "java",
    "sql", "postgresql", "mongodb", "redis", "kafka", "spark",
    "react", "typescript", "javascript", "node.js",
    "vllm", "tensorrt", "inference optimization", "quantization",
    "rag", "vector database", "qdrant", "pinecone", "weaviate",
    "networkx", "graph neural network", "gnn",
    "lightgbm", "xgboost", "random forest",
    "nlp", "computer vision", "multimodal",
    "leadership", "mentoring", "cross-functional",
    "research", "published", "neurips", "icml", "arxiv",
]


def parse_skills(text: str) -> list[str]:
    """Extract skills mentioned in resume text."""
    lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if skill in lower:
            found.append(skill)
    # also grab common patterns like "N YoE" or "X years"
    return found


def parse_title(text: str) -> Optional[str]:
    """Best-guess job title from the first few lines."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    title_patterns = [
        r"(senior|staff|principal|lead|junior|associate)?\s*(ml|machine learning|nlp|data|software|research|ai|backend|frontend|fullstack|full-stack|devops|mlops|quantitative)\s*(engineer|scientist|developer|researcher|analyst|architect|specialist)",
        r"(head|director|vp|manager|lead)\s+of\s+(engineering|ml|ai|data|platform)",
    ]
    for line in lines[:8]:
        lower = line.lower()
        for pat in title_patterns:
            m = re.search(pat, lower)
            if m:
                return line[:100]
    return None


def parse_location(text: str) -> Optional[str]:
    """Extract location hint from resume."""
    loc_patterns = [
        r"([A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*(?:CA|NY|TX|WA|IL|MA|GA|FL|CO|UK|DE|JP|UAE|MX|IN|AU|SG))",
        r"(Remote(?:\s*\(.*?\))?)",
    ]
    for pat in loc_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)[:100]
    return None


def parse_experience_years(text: str) -> float:
    """Rough estimate of total experience in years."""
    patterns = [
        r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:experience|exp)",
        r"(\d+)\s+YoE",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

    # Count year ranges like "2018 – 2023"
    years = re.findall(r"\b(20\d{2})\b", text)
    if len(years) >= 2:
        years_int = sorted([int(y) for y in years])
        span = years_int[-1] - years_int[0]
        return max(0, min(span, 20))
    return 3.0  # default assumption


def extract_emails_from_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)


def extract_phones_from_text(text: str) -> list[str]:
    # Match common phone formats
    return re.findall(r"(?:\+?\d[\d\s\-().]{7,}\d)", text)


def extract_names_from_text(text: str) -> list[str]:
    """Grab candidate-like names (2-3 capitalized words near top)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    name_pat = re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$")
    names = []
    for line in lines[:10]:
        m = name_pat.match(line)
        if m:
            names.append(m.group(1))
    return names


def parse_resume(path: str) -> dict:
    """Full parse of a resume file. Returns structured dict."""
    text = extract_text(path)
    if not text.strip():
        return {
            "text": "",
            "skills": [],
            "title": None,
            "location": None,
            "experience_years": 0,
            "emails": [],
            "phones": [],
            "names": [],
            "readable": False,
            "snippet": "",
        }

    return {
        "text": text,
        "skills": parse_skills(text),
        "title": parse_title(text),
        "location": parse_location(text),
        "experience_years": parse_experience_years(text),
        "emails": extract_emails_from_text(text),
        "phones": extract_phones_from_text(text),
        "names": extract_names_from_text(text),
        "readable": True,
        "snippet": text[:200].replace("\n", " ").strip(),
    }
