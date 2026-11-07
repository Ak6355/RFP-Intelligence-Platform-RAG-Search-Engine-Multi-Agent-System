"""CLI to run extraction, question answering, or ingestion through the agent graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from .orchestrator import build_agent_graph

load_dotenv()


def build_parser():
    parser = argparse.ArgumentParser(description="RFP Intelligence multi-agent system")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract all 20 fields for a bid and write JSON")
    extract.add_argument("--bid-id", required=True)
    extract.add_argument("--project-root", default=".")
    extract.add_argument("--max-retries", type=int, default=2)

    ask = sub.add_parser("ask", help="Ask a question about a bid with citations")
    ask.add_argument("--bid-id", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--project-root", default=".")

    ingest = sub.add_parser("ingest", help="Run Phase 2, 3 and 4 ingestion/indexing for current data")
    ingest.add_argument("--bid-id", required=True)
    ingest.add_argument("--project-root", default=".")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()
    graph = build_agent_graph(project_root=root)
    if args.command == "extract":
        state = graph.invoke({"mode": "extract", "bid_id": args.bid_id, "project_root": str(root), "max_retries": args.max_retries, "field_results": {}, "evidence_by_group": {}, "trace": []})
    elif args.command == "ask":
        state = graph.invoke({"mode": "qa", "bid_id": args.bid_id, "question": args.question, "project_root": str(root), "trace": []})
    else:
        state = graph.invoke({"mode": "ingest", "bid_id": args.bid_id, "project_root": str(root), "trace": []})

    if args.command == "extract":
        print(json.dumps({"status": "completed", "bid_id": args.bid_id, "output_json": state.get("final_json_path"), "validation": state.get("validation_report")}, indent=2, ensure_ascii=False))
    elif args.command == "ask":
        print(json.dumps({"bid_id": args.bid_id, "question": args.question, "answer": state.get("answer"), "citations": state.get("answer_sources", []), "confidence": state.get("confidence", 0.0)}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(state.get("ingestion_result", {}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
