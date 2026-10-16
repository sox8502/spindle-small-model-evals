"""Actual Modal spend of the spindle-evals deployment, grouped by Spindle config, from the hourly billing report."""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

import modal

FRONTEND = "spindle-evals"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True, type=datetime.fromisoformat, help="UTC ISO time")
    parser.add_argument("--until", type=datetime.fromisoformat, default=None, help="UTC ISO time")
    parser.add_argument("--env", default="main")
    args = parser.parse_args()

    # Billing is hourly and can lag by a day; partial trailing hours are excluded by Modal.
    report = modal.Workspace.from_context().billing.report(
        start=args.since, end=args.until, resolution="h"
    )
    by_app: dict[str, Decimal] = defaultdict(Decimal)
    by_resource: dict[str, Decimal] = defaultdict(Decimal)
    for item in report:
        name = item.description
        if item.environment_name == args.env and (name == FRONTEND or "-evals-" in name):
            by_app[name] += item.cost
            for resource, cost in item.cost_by_resource.items():
                by_resource[resource] += cost

    # Pool, inference and trainer apps are named spindle-<role>-<config>; the frontend is shared.
    by_config: dict[str, Decimal] = defaultdict(Decimal)
    for name, cost in by_app.items():
        by_config["shared-frontend" if name == FRONTEND else name.split("-", 2)[2]] += cost

    print(
        json.dumps(
            {
                "since": args.since.isoformat(),
                "until": args.until.isoformat() if args.until else None,
                "by_config_usd": {k: float(v) for k, v in sorted(by_config.items())},
                "by_app_usd": {k: float(v) for k, v in sorted(by_app.items())},
                "by_resource_usd": {k: float(v) for k, v in sorted(by_resource.items())},
                "total_usd": float(sum(by_app.values())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
