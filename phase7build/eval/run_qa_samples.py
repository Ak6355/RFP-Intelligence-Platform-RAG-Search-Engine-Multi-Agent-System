"""Run the 10 sample Q&A cases against a local API and save cited answers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default="eval/sample_qa_questions.json")
    parser.add_argument("--output", default="outputs/sample_qa_log.jsonl")
    args = parser.parse_args()
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(base_url=args.api, timeout=180.0) as client:
        for item in questions:
            response = client.post("/ask", json=item)
            response.raise_for_status()
            result = response.json()
            row = {**item, **result}
            rows.append(row)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Saved {len(rows)} cited answers to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
