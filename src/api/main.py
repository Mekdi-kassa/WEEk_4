"""FastAPI app for serving credit risk predictions."""

from fastapi import FastAPI

app = FastAPI(title="Credit Risk API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint for container/platform checks."""
    return {"status": "ok"}
