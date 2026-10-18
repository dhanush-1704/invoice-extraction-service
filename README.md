# Invoice Extraction Service

FastAPI `POST /extract` endpoint that extracts vendor, total amount, currency, and due date into a Pydantic-validated response.

## Request

```json
{"text":"Vendor: Acme Industries Ltd.\nCurrency: USD\nTotal Due: USD 123.45\nDue Date: 2026-11-15"}
```

## Response

```json
{"vendor":"Acme Industries Ltd","amount":123.45,"currency":"USD","date":"2026-11-15"}
```

Empty or unparseable text returns HTTP 422, not HTTP 500.

## Deploy

Import this repository into Vercel. Optionally configure `OLLAMA_BASE_URL` and `OLLAMA_MODEL` if you want the service to call an Ollama-compatible `/api/generate` endpoint for missing fields. The deterministic parser handles explicit invoice labels without requiring the local model to be online.
