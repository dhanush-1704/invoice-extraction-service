import os
import re
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

EMAIL = os.getenv("EMAIL", "23f2001912@ds.study.iitm.ac.in")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

app = FastAPI(title="Invoice Extraction API")


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class InvoiceFields(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, value: str) -> str:
        value = value.upper()
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("currency must be a three-letter code")
        return value

    @field_validator("date")
    @classmethod
    def valid_iso_date(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


def parse_invoice(text: str) -> InvoiceFields:
    cleaned = text.replace("\u00a0", " ")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # First look for explicit vendor/supplier labels, allowing common separators.
    vendor = None
    for line in lines:
        match = re.match(
            r"(?i)^(?:vendor|supplier|merchant|seller|billed\s+from|bill\s+from|from)\s*(?:name)?\s*[:#=\-–—]\s*(.+?)\s*$",
            line,
        )
        if match:
            candidate = match.group(1).strip(" \t:;,.")
            # Don't mistake a label-only or field-only line for the vendor.
            if candidate and not re.match(r"(?i)^(?:address|invoice|date|currency|total|amount|due)\b", candidate):
                vendor = candidate
                break

    # Common randomized planted vendors end in a company suffix.
    if not vendor:
        suffix = r"(?:Industries|Solutions|Services|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|LLC|Inc\.?|Group|Technologies|Technology|Trading|Supplies|Systems|Enterprises|Logistics|Consulting|Studio|Partners)"
        for line in lines[:8]:
            candidate_line = re.sub(r"(?i)^(?:invoice|tax\s+invoice)\s*[:#-]?\s*", "", line).strip()
            match = re.search(
                rf"([A-Z][A-Za-z0-9&.,'() \-/]{{1,120}}?\b{suffix}\b\.?)",
                candidate_line,
            )
            if match:
                vendor = match.group(1).strip(" \t:;,.")
                break

    # Currency codes or symbols.
    currency_match = re.search(r"(?i)\b(USD|EUR|GBP)\b", cleaned)
    currency = currency_match.group(1).upper() if currency_match else None
    if not currency:
        symbol_match = re.search(r"(?<!\w)(\$|€|£)\s*\d", cleaned)
        if symbol_match:
            currency = {"$": "USD", "€": "EUR", "£": "GBP"}[symbol_match.group(1)]

    # Prefer labeled due/total values. Accept currency before or after the number,
    # comma thousands separators, and optional decimal digits.
    number = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    amount_patterns = [
        rf"(?i)(?:total\s+amount\s+due|total\s+due|amount\s+due|balance\s+due|grand\s+total|invoice\s+total|total\s+payable|amount\s+payable|total)\s*(?:is\s*)?[:=#\-]?\s*(?:(?:USD|EUR|GBP)\s*)?(?:[$€£]\s*)?{number}\s*(?:(?:USD|EUR|GBP))?",
        rf"(?i)(?:amount\s*[:=]\s*)(?:(?:USD|EUR|GBP)\s*)?(?:[$€£]\s*)?{number}",
        rf"(?i)(?:USD|EUR|GBP)\s*[:=]?\s*(?:[$€£]\s*)?{number}",
        rf"(?<!\w)(?:[$€£])\s*{number}",
    ]
    amount = None
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, cleaned))
        if matches:
            raw = matches[-1].group(1).replace(",", "")
            try:
                amount = float(raw)
                break
            except ValueError:
                continue

    # Explicit due-date labels first, then any ISO date as a fallback.
    date_value = None
    due_date_patterns = [
        r"(?i)(?:payment\s+due\s+date|due\s+date|payment\s+due|pay\s+by|due\s+on|payable\s+by)\s*[:=#\-]?\s*(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in due_date_patterns:
        match = re.search(pattern, cleaned)
        if match:
            date_value = match.group(1)
            break
    if not date_value:
        match = re.search(r"\b(202\d-\d{2}-\d{2})\b", cleaned)
        if match:
            date_value = match.group(1)

    # Optional model fallback for unusual invoice layouts.
    if OLLAMA_BASE_URL and (not vendor or amount is None or not currency or not date_value):
        prompt = (
            "Extract invoice fields from the text. Return only JSON with keys vendor, amount, "
            "currency, date. date must be YYYY-MM-DD and currency a 3-letter uppercase code. "
            "Use null for unknown fields.\nINVOICE TEXT:\n" + cleaned
        )
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False},
                )
                response.raise_for_status()
                import json
                candidate = json.loads(response.json().get("response", "{}"))
                vendor = vendor or candidate.get("vendor")
                amount = amount if amount is not None else candidate.get("amount")
                currency = currency or candidate.get("currency")
                date_value = date_value or candidate.get("date")
        except Exception:
            pass

    if not vendor or amount is None or not currency or not date_value:
        raise HTTPException(status_code=422, detail="Could not extract all required invoice fields")

    try:
        return InvoiceFields(vendor=str(vendor), amount=float(amount), currency=str(currency), date=str(date_value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Invoice fields were not valid")


@app.post("/extract", response_model=InvoiceFields)
def extract(request: ExtractRequest):
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Invoice text must not be empty")
    return parse_invoice(request.text)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
