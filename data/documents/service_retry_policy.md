# Service Retry Policy

Retries can improve resilience when failures are temporary.

## Retryable Errors

Examples include:

- Temporary network failures
- HTTP 429 responses
- HTTP 503 responses
- Transient database connection errors

## Non-Retryable Errors

Examples include:

- Authentication failures
- Invalid requests
- Authorization failures
- Business validation errors

## Retry Strategy

Use exponential backoff and introduce jitter to avoid synchronized retries.

Retry attempts must be bounded.

Operations should be idempotent when retries may cause duplicate requests.
