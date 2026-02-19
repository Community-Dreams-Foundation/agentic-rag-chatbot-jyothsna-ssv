from pathlib import Path
import re


USER_MEMORY_FILE = Path("USER_MEMORY.md")
COMPANY_MEMORY_FILE = Path("COMPANY_MEMORY.md")

CONFIDENCE_THRESHOLD = 0.75

SENSITIVE_PATTERNS = (
    "password", "secret", "ssn", "api_key", "apikey", "token", "credential",
    "social security", "credit card", "pin ", "pii",
)


def _looks_sensitive(summary: str) -> bool:
    """Block saving anything that looks like passwords, keys, or other secrets."""
    lower = summary.lower()
    return any(p in lower for p in SENSITIVE_PATTERNS)


def _normalize_input(s: str) -> str:
    """Strip BOM/invisible chars, normalize punctuation and whitespace so regex matches reliably."""
    if not s:
        return ""
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = s.replace("\uff0e", ".").replace("\uff0c", ",")
    return " ".join(s.split())


def analyze_memory_signal(user_input: str):
    """Scan text for "I am a...", "I prefer...", "Our team..." and return what to save and where."""
    text = _normalize_input(user_input.strip())
    lower = text.lower()

    decisions = []

    role_match = re.search(r"\b(?:i\s+am|i'm)\s+a[n]?\s+(.+?)\s*(?:\.|,|\s+and\s+|$)", lower, re.IGNORECASE)
    if role_match:
        role = role_match.group(1).strip().rstrip(".,")
        if role:
            article = "an" if role[0].lower() in "aeiou" else "a"
            decisions.append({
                "should_write": True,
                "target": "USER",
                "summary": f"User is {article} {role}",
                "confidence": 0.9
            })
    else:
        role_short = re.search(r"\b(?:i\s+am|i'm)\s+(?!a[n]?\s)(.+?)\s*(?:\.|,|\s+and\s+|$)", lower, re.IGNORECASE)
        if role_short:
            role = text[role_short.start(1):role_short.end(1)].strip().rstrip(".,")
            if role and len(role) >= 2:
                article = "an" if role[0].lower() in "aeiou" else "a"
                decisions.append({
                    "should_write": True,
                    "target": "USER",
                    "summary": f"User is {article} {role}",
                    "confidence": 0.9
                })

    preference_match = re.search(r"\b(?:i\s+prefer|i'd\s+prefer)\s+(.+?)\s*(?:\.|,|\s+and\s+|$)", lower, re.IGNORECASE)
    if preference_match:
        preference = preference_match.group(1).strip().rstrip(".,")
        if preference:
            preference_norm = preference.replace(" instead of ", " over ")
            decisions.append({
                "should_write": True,
                "target": "USER",
                "summary": f"User prefers {preference_norm}",
                "confidence": 0.85
            })

    org_match = re.search(r"\bour\s+team\s+(.+?)\s*(?:\.|,|$)", lower, re.IGNORECASE)
    if org_match:
        insight = text[org_match.start(1):org_match.end(1)].strip().rstrip(".,")
        if insight:
            decisions.append({
                "should_write": True,
                "target": "COMPANY",
                "summary": f"The team {insight}",
                "confidence": 0.8
            })

    weak_match = re.search(r"\b(?:i\s+might|i\s+may|i'm\s+thinking|i\s+could)\s+.+", lower)
    if weak_match and not decisions:
        decisions.append({
            "should_write": False,
            "target": "USER",
            "summary": "",
            "confidence": 0.5
        })

    if not decisions and text and not _looks_sensitive(text):
        decisions.append({
            "should_write": True,
            "target": "USER",
            "summary": f"User note: {text}",
            "confidence": 0.8
        })

    return decisions


def persist_memory(decisions):
    """Write approved decisions to USER_MEMORY.md or COMPANY_MEMORY.md, skip dupes and low confidence."""
    written = []

    for decision in decisions:
        if not decision.get("should_write"):
            continue

        if decision.get("confidence", 0) < CONFIDENCE_THRESHOLD:
            continue

        target = decision["target"]
        summary = decision["summary"]

        if _looks_sensitive(summary):
            continue

        if target == "USER":
            file_path = USER_MEMORY_FILE
        elif target == "COMPANY":
            file_path = COMPANY_MEMORY_FILE
        else:
            continue

        entry = f"- {summary}\n"

        if not file_path.exists():
            file_path.write_text("# Memory Log\n\n", encoding="utf-8")

        existing = file_path.read_text(encoding="utf-8")

        if summary not in existing:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(entry)

            written.append({
                "target": target,
                "summary": summary,
                "confidence": decision.get("confidence", 0.0)
            })

    return written
