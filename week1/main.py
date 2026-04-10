from datetime import datetime, timezone
from typing import Literal, cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="AI Engineering Bootcamp - Assignment 1",
    description="FastAPI app with health check, text summarization, and sentiment analysis endpoints.",
    version="1.0.0",
)

MODEL = "gpt-4o-mini"
client = OpenAI()
Sentiment = Literal["positive", "negative", "neutral"]
SENTIMENT_LABELS: tuple[Sentiment, ...] = ("positive", "negative", "neutral")

# --- Request / Response Models ---


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_length: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Approximate max length of summary in words",
    )


class SummarizeResponse(BaseModel):
    summary: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")


class SentimentResponse(BaseModel):
    sentiment: Sentiment
    confidence: float
    explanation: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# --- Prompt Helpers ---

SUMMARIZE_PROMPT = (
    "You are a precise summarization assistant. "
    "Summarize the following text in at most {max_length} words. "
    "Return ONLY the summary, no preamble.\n\n"
    "Text:\n{text}"
)

SENTIMENT_PROMPT = (
    "Analyze the sentiment of the following text. "
    "Respond in EXACTLY this format (3 lines, no extra text):\n"
    "Sentiment: <positive|negative|neutral>\n"
    "Confidence: <0.0 to 1.0>\n"
    "Explanation: <one sentence explaining why>\n\n"
    "Text:\n{text}"
)

def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("LLM returned an empty response.")
    return content


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="text must not be blank")
    return cleaned


def _parse_sentiment(raw: str) -> SentimentResponse:
    fields: dict[str, str] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not key or not value:
            raise ValueError("Sentiment response format was invalid.")
        if key not in {"sentiment", "confidence", "explanation"}:
            raise ValueError("Sentiment response contained unexpected fields.")
        fields[key] = value

    if set(fields) != {"sentiment", "confidence", "explanation"}:
        raise ValueError("Sentiment response was missing required fields.")

    sentiment = fields["sentiment"].lower()
    if sentiment not in SENTIMENT_LABELS:
        raise ValueError("Sentiment label was invalid.")

    try:
        confidence = float(fields["confidence"])
    except ValueError as exc:
        raise ValueError("Confidence score was not a number.") from exc

    return SentimentResponse(
        sentiment=cast(Sentiment, sentiment),
        confidence=round(confidence, 2),
        explanation=fields["explanation"],
    )


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        prompt = SUMMARIZE_PROMPT.format(
            text=_clean_text(request.text), max_length=request.max_length
        )
        summary = _call_llm(prompt, max_tokens=request.max_length * 2)
        return SummarizeResponse(summary=summary)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid LLM response: {exc}")
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")


@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    try:
        prompt = SENTIMENT_PROMPT.format(text=_clean_text(request.text))
        raw = _call_llm(prompt, max_tokens=150)
        return _parse_sentiment(raw)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid LLM response: {exc}")
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}")
