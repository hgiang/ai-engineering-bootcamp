# AI Engineering Bootcamp - Assignment 1

FastAPI application with health check, text summarization, and sentiment analysis endpoints, powered by OpenAI (`gpt-4o-mini`).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check - returns status and timestamp |
| POST | `/summarize` | Summarize text with configurable max length |
| POST | `/analyze-sentiment` | Analyze sentiment with confidence score |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Run locally

```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

## Run prompt experiments

```bash
python prompt_experiments.py
```

## Deploy to Render

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Connect your GitHub repo
4. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `OPENAI_API_KEY` = your key
