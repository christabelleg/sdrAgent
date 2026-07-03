import argparse
import json
import os
from typing import Any, Dict, List

from sdr_pipeline_agent import run_daily_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SDR recommendations and write them to Notion."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("TOP_N", "10")),
        help="How many top recommendations to generate and publish.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate recommendations without writing to Notion.",
    )
    return parser.parse_args()


def compact_view(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "company": rec.get("company"),
            "contact_name": rec.get("contact_name"),
            "priority": rec.get("priority"),
            "next_best_action": rec.get("next_best_action"),
        }
        for rec in recommendations
    ]


def main() -> None:
    args = parse_args()
    recommendations = run_daily_review(limit=args.limit, dry_run=args.dry_run)

    print(f"Generated {len(recommendations)} recommendations. dry_run={args.dry_run}")
    print(json.dumps(compact_view(recommendations), indent=2))


if __name__ == "__main__":
    main()
