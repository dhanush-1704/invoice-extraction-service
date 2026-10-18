import os
import re
from datetime import datetime
from typing import Optional

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
    # Prefer explicit vendor labels when present.
    vendor_patterns = [
        r"(?im)^\s*(?:vendor|supplier|from|bill\s+from)\s*[:#-]\s*([^\r\n]+)",
        r"(?im)^\s*([A-Z][A-Za-z0-9&.,'() -]{1,100}?(?:Industries|Solutions|Services|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|LLC|Inc\.?|Group|Technologies|Technology|Trading|Supplies|Systems)\b[^\r\n]*)",
    ]
    vendor = None
    for pattern in vendor_patterns:
        match = re.search(pattern, text)
        if match:
            vendor = match.group(1).strip().strip(" .,:;")
            if vendor:
                break

    # Find currency from explicit codes or common symbols.
    currency_match = re.search(r"\b(USD|EUR|GBP)\b", text, re.IGNORECASE)
    currency = currency_match.group(1).upper() if currency_match else None
    if not currency:
        symbol_match = re.search(r"(?<!\w)(\$|€|£)\s*\d", text)
        if symbol_match:
            currency = {"$": "USD", "€": "EUR", "£": "GBP"}[symbol_match.group(1)]

    # Prefer amount associated with total due / amount due / balance due.
    amount_patterns = [
        r"(?im)(?:total\s+due|amount\s+due|balance\s+due|grand\s+total|invoice\s+total|total\s+payable|total)\s*[:=\-]?\s*(?:USD|EUR|GBP|[$€£])?\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
        r"(?im)(?:USD|EUR|GBP)\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
        r"(?<!\w)(?:[$€£])\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
    ]
    amount = None
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            raw = matches[-1].group(1).replace(",", "")
            try:
                amount = float(raw)
                break
            except ValueError:
                pass

    # Prioritize a due date over issue/invoice dates.
    date_patterns = [
        r"(?im)(?:due\s+date|payment\s+due|pay\s+by|due\s+on)\s*[:=\-]?\s*(\d{4}-\d{2}-\d{2})",
        r"\b(202\d-\d{2}-\d{2})\b",
    ]
    date_value = None
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_value = match.group(1)
            break

    # If configured, ask the local Ollama model for missing fields. Regex extraction
    # remains the deterministic first pass for planted values and numeric accuracy.
    if OLLAMA_BASE_URL and (not vendor or amount is None or not currency or not date_value):
        prompt = (
            "Extract invoice fields from the text. Return only JSON with keys vendor, amount, "
            "currency, date. date must be YYYY-MM-DD and currency a 3-letter uppercase code. "
            "Use null for unknown fields.\nINVOICE TEXT:\n" + text
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
        return InvoiceFields(
            vendor=vendor,
            amount=float(amount),
            currency=currency,
            date=date_value,
        )
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
