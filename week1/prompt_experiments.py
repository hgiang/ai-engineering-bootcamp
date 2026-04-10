"""
Prompt Engineering Experiments
==============================
Tests 3 prompt variations for /summarize and /analyze-sentiment,
prints results for documentation.
"""

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

SAMPLE_TEXT_SUMMARIZE = (
    "Artificial intelligence has transformed numerous industries over the past decade. "
    "In healthcare, AI systems now assist doctors in diagnosing diseases from medical "
    "images with accuracy rivaling human specialists. In finance, machine learning "
    "algorithms detect fraudulent transactions in real-time, saving billions of dollars "
    "annually. The transportation sector has seen autonomous vehicles move from science "
    "fiction to reality, with companies testing self-driving cars on public roads. "
    "Education has been revolutionized by adaptive learning platforms that personalize "
    "instruction for each student. However, these advances come with challenges: "
    "concerns about job displacement, algorithmic bias, privacy, and the need for "
    "robust regulation. As AI continues to evolve, society must balance innovation "
    "with responsible development to ensure these technologies benefit everyone."
)

SAMPLE_TEXT_SENTIMENT_POS = "I absolutely love this product! It exceeded all my expectations and the customer service was phenomenal."
SAMPLE_TEXT_SENTIMENT_NEG = "This was the worst experience I've ever had. The product broke within a day and nobody responded to my complaint."
SAMPLE_TEXT_SENTIMENT_NEU = "The package arrived on Tuesday. It contained the items listed on the invoice."

# --- Summarization Prompt Variations ---

SUMMARIZE_V1_BASIC = (
    "Summarize this text:\n\n{text}"
)

SUMMARIZE_V2_CONSTRAINED = (
    "You are a precise summarization assistant. "
    "Summarize the following text in at most {max_length} words. "
    "Return ONLY the summary, no preamble.\n\n"
    "Text:\n{text}"
)

SUMMARIZE_V3_COT = (
    "Read the following text carefully. "
    "First, identify the 3 most important points. "
    "Then, combine them into a concise summary of at most {max_length} words. "
    "Output ONLY the final summary.\n\n"
    "Text:\n{text}"
)

# --- Sentiment Prompt Variations ---

SENTIMENT_V1_BASIC = (
    "What is the sentiment of this text? Is it positive, negative, or neutral?\n\n{text}"
)

SENTIMENT_V2_STRUCTURED = (
    "Analyze the sentiment of the following text. "
    "Respond in EXACTLY this format (3 lines, no extra text):\n"
    "Sentiment: <positive|negative|neutral>\n"
    "Confidence: <0.0 to 1.0>\n"
    "Explanation: <one sentence explaining why>\n\n"
    "Text:\n{text}"
)

SENTIMENT_V3_FEWSHOT = (
    "Analyze sentiment. Respond in exactly 3 lines.\n\n"
    "Example 1:\n"
    "Text: \"The sunset was beautiful and I felt at peace.\"\n"
    "Sentiment: positive\n"
    "Confidence: 0.92\n"
    "Explanation: Expresses beauty and contentment.\n\n"
    "Example 2:\n"
    "Text: \"The meeting was moved to 3pm.\"\n"
    "Sentiment: neutral\n"
    "Confidence: 0.95\n"
    "Explanation: Purely factual statement with no emotional content.\n\n"
    "Now analyze:\n"
    "Text: \"{text}\"\n"
    "Sentiment:"
)


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def run_experiments():
    print("=" * 70)
    print("SUMMARIZATION EXPERIMENTS")
    print("=" * 70)

    max_length = 50
    variations = [
        ("V1 - Basic (no instructions)", SUMMARIZE_V1_BASIC),
        ("V2 - Constrained with role (CHOSEN)", SUMMARIZE_V2_CONSTRAINED),
        ("V3 - Chain-of-Thought", SUMMARIZE_V3_COT),
    ]

    for name, template in variations:
        prompt = template.format(text=SAMPLE_TEXT_SUMMARIZE, max_length=max_length)
        result = call_llm(prompt, max_tokens=200)
        print(f"\n--- {name} ---")
        print(f"Prompt (first 120 chars): {prompt[:120]}...")
        print(f"Output: {result}")
        print(f"Word count: {len(result.split())}")

    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS EXPERIMENTS")
    print("=" * 70)

    sentiment_variations = [
        ("V1 - Basic", SENTIMENT_V1_BASIC),
        ("V2 - Structured output (CHOSEN)", SENTIMENT_V2_STRUCTURED),
        ("V3 - Few-shot examples", SENTIMENT_V3_FEWSHOT),
    ]

    test_texts = [
        ("Positive", SAMPLE_TEXT_SENTIMENT_POS),
        ("Negative", SAMPLE_TEXT_SENTIMENT_NEG),
        ("Neutral", SAMPLE_TEXT_SENTIMENT_NEU),
    ]

    for name, template in sentiment_variations:
        print(f"\n--- {name} ---")
        for label, text in test_texts:
            prompt = template.format(text=text)
            result = call_llm(prompt, max_tokens=150)
            print(f"\n  [{label}] Input: {text[:60]}...")
            print(f"  Output: {result}")


if __name__ == "__main__":
    run_experiments()
