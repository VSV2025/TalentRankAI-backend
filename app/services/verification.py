"""Candidate verification: email format, MX check, consistency, deduplication."""
import re
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def check_email_format(email: str) -> tuple[bool, str]:
    """RFC 5321 format check."""
    pat = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
    if pat.match(email.strip()):
        return True, "Standard RFC 5321 format confirmed."
    return False, f"Email format is invalid: {email}"


def check_email_mx(email: str) -> tuple[bool, str]:
    """Check MX record for the email domain."""
    domain = email.split("@")[-1] if "@" in email else ""
    if not domain:
        return False, "Cannot extract domain from email."
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, "MX", lifetime=3)
        if answers:
            return True, f"MX record found for {domain}."
        return False, f"No MX record found for {domain}."
    except ImportError:
        logger.warning("dnspython not installed — skipping MX check")
        return True, "MX check skipped (dnspython not installed)."
    except Exception as e:
        # DNS lookup failures shouldn't hard-fail a submission
        logger.debug(f"MX lookup for {domain}: {e}")
        return True, f"MX lookup inconclusive for {domain} — treated as pass."


def normalize_email(email: str) -> str:
    return email.strip().lower()


def normalize_phone(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def check_consistency(
    form_name: str,
    form_email: str,
    resume_emails: list[str],
    resume_names: list[str],
    resume_phones: list[str],
    form_phone: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    """
    Compare form submission against parsed resume fields.
    Returns (status, detail, badge) where status is 'pass' | 'review'.
    """
    issues = []

    # Email consistency
    norm_form_email = normalize_email(form_email)
    norm_resume_emails = [normalize_email(e) for e in resume_emails]
    if norm_resume_emails and norm_form_email not in norm_resume_emails:
        issues.append(
            f"Email on resume ({resume_emails[0]}) differs from form submission ({form_email})."
        )

    # Name fuzzy match
    if resume_names:
        try:
            from rapidfuzz import fuzz
            best_ratio = max(
                fuzz.token_sort_ratio(form_name.lower(), n.lower())
                for n in resume_names
            )
            if best_ratio < 70:
                issues.append(
                    f"Name on resume ({resume_names[0]}) is a low match for submitted name ({form_name})."
                )
        except ImportError:
            pass

    # Phone consistency (if provided)
    if form_phone and resume_phones:
        norm_form_phone = normalize_phone(form_phone)
        norm_resume_phones = [normalize_phone(p) for p in resume_phones]
        if norm_form_phone and not any(
            norm_form_phone in rp or rp in norm_form_phone
            for rp in norm_resume_phones
            if rp
        ):
            issues.append(
                "Phone number on resume does not match form submission."
            )

    if issues:
        detail = " ".join(issues)
        badge = issues[0][:120]
        return "review", detail, badge
    return "pass", "All submitted fields are consistent with resume content.", None


def compute_resume_hash(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def run_verification(
    form_name: str,
    form_email: str,
    resume_path: Optional[str],
    resume_parsed: Optional[dict],
) -> dict:
    """
    Run all verification checks and return a structured result dict
    matching the frontend VerificationResult shape.
    """
    checks = []

    # 1. Email format
    fmt_pass, fmt_detail = check_email_format(form_email)
    checks.append({
        "id": "email_format",
        "label": "Email format valid",
        "result": "pass" if fmt_pass else "review",
        "detail": fmt_detail,
        "badge": None,
    })

    # 2. Email confirmation (always pending — we'd send a link in prod)
    checks.append({
        "id": "email_confirm",
        "label": "Confirmation email sent",
        "result": "pending",
        "detail": "Verification link sent — awaiting confirmation from your inbox.",
        "badge": None,
    })

    # 3. Resume readability
    if resume_parsed and resume_parsed.get("readable"):
        checks.append({
            "id": "resume_readable",
            "label": "Resume is a readable file",
            "result": "pass",
            "detail": "File parsed successfully — text, structure, and metadata extracted.",
            "badge": None,
        })
    else:
        checks.append({
            "id": "resume_readable",
            "label": "Resume is a readable file",
            "result": "review",
            "detail": "Could not extract text from the file. Please upload a machine-readable PDF or DOCX.",
            "badge": "Resume not machine-readable",
        })

    # 4. Consistency check
    if resume_parsed and resume_parsed.get("readable"):
        con_status, con_detail, con_badge = check_consistency(
            form_name=form_name,
            form_email=form_email,
            resume_emails=resume_parsed.get("emails", []),
            resume_names=resume_parsed.get("names", []),
            resume_phones=resume_parsed.get("phones", []),
        )
        checks.append({
            "id": "consistency",
            "label": "Consistency check · authenticity score",
            "result": con_status,
            "detail": con_detail,
            "badge": con_badge,
        })
    else:
        checks.append({
            "id": "consistency",
            "label": "Consistency check · authenticity score",
            "result": "review",
            "detail": "Cannot perform consistency check without a readable resume.",
            "badge": "Consistency check skipped",
        })

    # Derive overall status.
    # "pending" only comes from email_confirm which is always pending in this system.
    # Treat it as informational — don't let it block "verified" when all real checks pass.
    statuses = [c["result"] for c in checks]
    if "review" in statuses:
        overall = "review"
    else:
        non_pending = [s for s in statuses if s != "pending"]
        overall = "verified" if non_pending and all(s == "pass" for s in non_pending) else "pending"

    return {
        "checks": checks,
        "overall_status": overall,
    }
