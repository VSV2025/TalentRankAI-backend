"""Resume text extraction and structured field parsing."""
import hashlib
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text(path: str) -> str:
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
        import fitz
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


# Comprehensive skill keyword list — ordered by specificity (longer phrases first to prevent partial match issues)
SKILL_KEYWORDS = [
    # LLM / NLP
    "large language model", "large language models", "natural language processing",
    "named entity recognition", "text classification", "sentence transformers",
    "distributed training", "inference optimization", "model serving",
    "fine-tuning", "finetuning", "pre-training", "pretraining",
    "reinforcement learning from human feedback", "rlhf", "rag",
    "retrieval augmented generation", "vector search", "semantic search",
    "hybrid search", "dense retrieval", "sparse retrieval",
    "graph neural network", "gnn",
    "computer vision", "multimodal",
    # Models / frameworks
    "pytorch", "tensorflow", "jax", "keras", "huggingface", "hugging face",
    "transformers", "bert", "gpt", "llama", "mistral", "falcon", "gemma",
    "langchain", "langgraph", "langraph", "llamaindex", "llama-index",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "random forest", "gradient boosting",
    "vllm", "tensorrt", "triton", "bentoml",
    "lora", "qlora", "peft",
    # Infrastructure / cloud
    "kubernetes", "docker", "airflow", "kubeflow", "mlflow",
    "mlops", "ci/cd", "sagemaker", "vertex ai",
    "aws", "gcp", "azure", "lambda", "ec2", "s3",
    # Vector DBs
    "qdrant", "pinecone", "weaviate", "milvus", "faiss", "annoy",
    "elasticsearch", "opensearch", "solr",
    # Data
    "apache spark", "kafka", "spark", "postgresql", "mongodb",
    "redis", "snowflake", "bigquery", "dbt",
    "pandas", "numpy",
    # Languages
    "python", "c++", "cuda", "rust", "go", "golang", "java",
    "typescript", "javascript", "react", "node.js",
    "sql",
    # Soft / other
    "leadership", "mentoring", "cross-functional",
    "quantization", "distillation",
    "a/b testing", "experimentation",
    "published", "neurips", "icml", "arxiv",
    # Shorter patterns last (prevent over-matching)
    "nlp", "llm", "llms", "gpt", "bert", "rag",
    "ml", "ai",
]

# De-duplicate while preserving order
_seen = set()
_DEDUPED_SKILLS = []
for _s in SKILL_KEYWORDS:
    if _s not in _seen:
        _seen.add(_s)
        _DEDUPED_SKILLS.append(_s)
SKILL_KEYWORDS = _DEDUPED_SKILLS


def parse_skills(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        # Use word-boundary matching for short/ambiguous skills
        if len(skill) <= 4:
            if re.search(r'\b' + re.escape(skill) + r'\b', lower):
                found.append(skill)
        elif skill in lower:
            found.append(skill)
    return found


# Title patterns — ordered from most to least specific
_TITLE_PATTERNS = [
    # "Head/Director/VP of Engineering/ML/AI"
    r"(head|director|vp|vice president)\s+of\s+(engineering|ml|ai|machine learning|data|platform|product)",
    # Compound title: seniority + domain + role
    r"(senior|staff|principal|lead|junior|associate|founding)?\s*(ml|machine learning|nlp|natural language|data|software|research|ai|backend|frontend|fullstack|full.stack|devops|mlops|quantitative|inference|platform)\s*(engineer|scientist|developer|researcher|analyst|architect|specialist|lead|manager)",
    # Simple: AI/ML/Data + role
    r"(ai|machine learning|ml|data science|nlp|deep learning|computer vision)\s*(engineer|scientist|researcher|developer)",
    # MLOps / Platform
    r"(mlops|ml platform|ml infrastructure|model)\s+(engineer|lead|architect|specialist)",
    # Research variants
    r"(research)\s+(engineer|scientist|lead)",
    # Software engineer variations
    r"(software|backend|frontend|fullstack|full.stack|platform)\s+(engineer|developer|architect)",
]


def parse_title(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Search first 15 lines for a title pattern
    for line in lines[:15]:
        lower = line.lower()
        for pat in _TITLE_PATTERNS:
            m = re.search(pat, lower)
            if m:
                # Return only the matched title portion (not the full line which may contain "at Company, 2018-present")
                matched = line[m.start():m.end()].strip()
                return matched[:120]
    return None


# Location patterns — expanded to cover more formats
_LOCATION_PATTERNS = [
    # "City, State/Country abbreviation" — US and international
    r"([A-Z][a-zA-Z\s]+,\s*(?:CA|NY|TX|WA|IL|MA|GA|FL|CO|NC|VA|OH|AZ|NJ|OR|MN|IN|UK|DE|JP|UAE|MX|IN|AU|SG|NZ|IE|PK|BD|LK|NL|FR|ES|IT|SE|NO|DK|FI|CH|PL|CZ|AT|BE|PT|RO|HU|GR))\b",
    # "City, Country (full)" — common international cities
    r"([A-Z][a-zA-Z]+(?:[\s\-][A-Z][a-zA-Z]+)?,\s*(?:India|United States|United Kingdom|Canada|Australia|Germany|France|Singapore|Japan|UAE|Netherlands|Sweden|Switzerland))",
    # "Remote (Location)" or "Remote - Location"
    r"(Remote(?:\s*[\(\-\/]\s*[A-Z][a-zA-Z\s,]+[\)\-]?)?)",
    # Bangalore / Bengaluru / Pune / Mumbai etc. alone on a line or near top
    r"\b(Bangalore|Bengaluru|Mumbai|Pune|Delhi|Hyderabad|Chennai|Noida|Gurgaon|Gurugram|Kolkata|Ahmedabad)\b",
    # "London, UK" etc.
    r"([A-Z][a-zA-Z]+,\s*[A-Z]{2,3})",
]


def parse_location(text: str) -> Optional[str]:
    for pat in _LOCATION_PATTERNS:
        m = re.search(pat, text)
        if m:
            loc = m.group(1).strip()
            if len(loc) >= 3:
                return loc[:120]
    return None


def parse_experience_years(text: str) -> float:
    # Explicit "X years of experience" or "X YoE"
    explicit_patterns = [
        r"(\d+)\+?\s+years?\s+(?:of\s+)?(?:total\s+)?(?:professional\s+)?(?:work\s+)?(?:experience|exp)",
        r"(\d+)\s+YoE",
        r"(\d+)\s+years?\s+(?:in|at|of\s+work)",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0 < val <= 40:
                    return val
            except ValueError:
                pass

    # Count distinct year mentions in work history (date ranges)
    year_mentions = re.findall(r"\b(20\d{2}|19\d{2})\b", text)
    if len(year_mentions) >= 2:
        years_int = sorted(set(int(y) for y in year_mentions))
        span = years_int[-1] - years_int[0]
        if 1 <= span <= 30:
            return float(span)

    return 0.0  # no parseable experience data — lets the review table flag this correctly


def extract_emails_from_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)


def extract_phones_from_text(text: str) -> list[str]:
    """
    Multi-pattern phone extractor.  Handles:
    - International  : +91 9876543210, +1 (555) 123-4567
    - US with parens : (555) 123-4567
    - Dashed/dotted  : 555-123-4567, 555.123.4567
    - Indian mobile  : 9876543210, 98765 43210  (starts 6-9)
    - Bare runs      : 10-15 consecutive digits as last-resort
    """
    _patterns = [
        # International prefix +XX... (covers +91, +1, +44 etc.)
        r"\+\d{1,3}[\s\-\.]?\(?\d?\)?[\s\-\.]?\d{1,5}[\s\-\.]?\d{2,5}[\s\-\.]?\d{2,9}",
        # US/intl with parenthesized area code: (555) 123-4567
        r"\(\d{3,5}\)[\s\-\.]?\d{3,5}[\s\-\.]?\d{3,6}",
        # Generic with separator: 555-123-4567, 555.123.4567, 555 123 4567
        r"\b\d{3}[\s\-\.]\d{3,4}[\s\-\.]\d{4,6}\b",
        # Indian 10-digit mobile (starts 6-9, optional mid-space)
        r"\b[6-9]\d{4}[\s\-]?\d{5}\b",
        # Bare 10-15 digit runs (last resort — catches anything remaining)
        r"\b\d{10,15}\b",
    ]
    found: list[str] = []
    seen_digits: set[str] = set()
    for pat in _patterns:
        for m in re.finditer(pat, text):
            raw = m.group().strip()
            digits = re.sub(r"\D", "", raw)
            # Skip if too short or looks like a 4-digit year block
            if len(digits) < 7 or re.fullmatch(r"(19|20)\d{2}", digits):
                continue
            if digits not in seen_digits:
                seen_digits.add(digits)
                found.append(raw)
    return found


def extract_names_from_text(text: str) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Match lines that look like "Firstname Lastname" (2-4 capitalized words, no other text)
    name_pat = re.compile(r"^([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,3})$")
    names = []
    for line in lines[:12]:
        m = name_pat.match(line.strip())
        if m:
            candidate_name = m.group(1)
            # Filter out false positives (common section headers etc.)
            skip_words = {"Summary", "Experience", "Education", "Skills", "Projects",
                          "Contact", "Profile", "References", "Publications", "Awards",
                          "Certifications", "Work Experience", "Career", "Professional"}
            if candidate_name not in skip_words:
                names.append(candidate_name)
    return names


def parse_resume(path: str) -> dict:
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
        "snippet": text[:300].replace("\n", " ").strip(),
    }
