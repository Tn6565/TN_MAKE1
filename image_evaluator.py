"""
image_evaluator.py
- Uses OpenAI v1 client to perform a structured, "辛口" 5-item evaluation:
  1. Quality
  2. Color
  3. Originality
  4. Immediate impact (first impression)
  5. Memorability
- Produces numeric scores (0-100), commentary, and a sellability score (0-100).
- Saves evaluations to CSV and appends to a learning JSON for reuse.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Do not crash here — caller may handle missing key
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

LEARNING_FILE = "learning_image.json"
EVAL_CSV = "evaluation_results.csv"
os.makedirs(os.path.dirname(LEARNING_FILE) or ".", exist_ok=True)
if not os.path.exists(LEARNING_FILE):
    with open(LEARNING_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)


def _parse_scores_from_text(text: str) -> Dict[str, int] | None:
    """
    Try to extract 5 integer scores from LLM text.
    The model is asked to output JSON, but we implement a robust fallback.
    """
    # Try to find a JSON blob
    import re
    try:
        # find first JSON-like {...}
        m = re.search(r"(\{[\s\S]*?\})", text)
        if m:
            candidate = m.group(1)
            data = json.loads(candidate)
            if isinstance(data, dict) and all(k in data for k in ["quality","color","originality","impact","memorability"]):
                return {k: int(data[k]) for k in data}
    except Exception:
        pass

    # fallback: find numbers in order
    nums = re.findall(r"(\d{1,3})", text)
    if len(nums) >= 5:
        vals = list(map(int, nums[:5]))
        return {
            "quality": vals[0],
            "color": vals[1],
            "originality": vals[2],
            "impact": vals[3],
            "memorability": vals[4]
        }
    return None


def evaluate_image_with_ai(image_path: str, prompt_text: str) -> Tuple[Dict[str, int], str, int]:
    """
    Returns (scores_dict, comment_text, sellability_score)
    """
    global client
    if client is None:
        # fallback trivial evaluation if no API key
        scores = {"quality": 70, "color": 70, "originality": 60, "impact": 65, "memorability": 60}
        comment = "OpenAI API key not set: returning default mock evaluation."
        sellability = calculate_sellability(scores)
        return scores, comment, sellability

    system = "You are a critical, experienced image evaluator for stock marketplaces. Provide JSON with numeric scores (0-100)."
    user = (
        "Evaluate the image at path: " + image_path + "\n"
        "Prompt / market note: " + prompt_text + "\n\n"
        "Return a JSON object exactly like:\n"
        '{ "quality": int, "color": int, "originality": int, "impact": int, "memorability": int, "comment":"..." }\n'
        "Be strict: prefer conservative scores if unsure."
    )

    # v1 client chat completion style
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=300
    )

    text = resp.choices[0].message.content
    parsed = _parse_scores_from_text(text)
    if parsed is None:
        # return fallback minimal parsing
        parsed = {"quality": 60, "color": 60, "originality": 50, "impact": 55, "memorability": 50}
    comment = text
    sellability = calculate_sellability(parsed)
    return parsed, comment, sellability


def calculate_sellability(scores: Dict[str, int]) -> int:
    # weighted formula — can be tuned
    w_quality = 0.25
    w_color = 0.20
    w_originality = 0.25
    w_impact = 0.20
    w_mem = 0.10
    total = (
        w_quality * scores.get("quality", 0) +
        w_color * scores.get("color", 0) +
        w_originality * scores.get("originality", 0) +
        w_impact * scores.get("impact", 0) +
        w_mem * scores.get("memorability", 0)
    )
    return int(max(0, min(100, round(total))))

def append_learning(record: dict):
    # Append the evaluation record to learning json
    with open(LEARNING_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.append(record)
    with open(LEARNING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_results_csv(rows: list):
    """
    rows: list of dicts. Will append to evaluation CSV or create it.
    """
    headers = None
    if os.path.exists(EVAL_CSV):
        with open(EVAL_CSV, "r", encoding="utf-8", newline="") as f:
            import csv as _csv
            reader = _csv.reader(f)
            headers = next(reader, None)
    else:
        headers = None

    with open(EVAL_CSV, "a", encoding="utf-8", newline="") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames = rows[0].keys())
        if f.tell() == 0:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return EVAL_CSV
