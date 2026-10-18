"""Command-line entry point.

    python -m aml.run_pipeline                       # synthetic book, default thresholds
    python -m aml.run_pipeline --sweep large_cash    # tune one threshold
    python -m aml.run_pipeline --accounts 20000 --days 365

Writes into ``aml/results/``:

* ``monitoring_report.md`` - per-rule table, portfolio summary, missed-case
  analysis, and the threshold sweep.
* ``alerts.csv`` - the alert queue itself, as an analyst would receive it.
* ``monitoring.sqlite`` - transactions, accounts, alerts, and SQL analytics views.
* ``sweep_<field>.csv`` and ``sweep_<field>.png`` - the tuning evidence.
"""

from __future__ import annotations

import argparse
import os
from datetime import date

from .data import SANCTIONED_COUNTERPARTIES, BookConfig, generate_book
from .evaluate import (
    CostModel,
    missed_case_analysis,
    per_rule_table,
    portfolio_summary,
    threshold_sweep,
)
from .report import build_dashboard_html, cost_curve_plot, summary_markdown, to_markdown
from .rules import Thresholds, run_all_rules
from .sql_analytics import write_analytics_database

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PER_RULE_COLS = [
    "rule",
    "alerts",
    "tp",
    "fp",
    "precision",
    "recall",
    "target_recall",
    "target_population",
    "alerts_per_true_case",
]
SWEEP_COLS = [
    "accounts_reviewed",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "missed_sanctions_cases",
    "analyst_hours",
    "false_positive_cost_eur",
    "missed_case_cost_eur",
    "total_expected_cost_eur",
]

SWEEP_VALUES = {
    "large_cash": [5_000, 7_500, 10_000, 15_000, 25_000],
    "structuring_min_count": [2, 3, 4, 5],
    "structuring_band_low": [0.5, 0.65, 0.8, 0.9],
    "pass_through_ratio": [0.70, 0.80, 0.85, 0.90, 0.95],
    "dormant_days": [30, 45, 60, 90, 120],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounts", type=int, default=2_000)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20_260_101)
    parser.add_argument(
        "--sweep",
        choices=sorted(SWEEP_VALUES),
        default="large_cash",
        help="which threshold to sweep for the tuning table",
    )
    parser.add_argument("--minutes-per-alert", type=float, default=20.0)
    parser.add_argument("--cost-per-missed-case", type=float, default=25_000.0)
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    config = BookConfig(n_accounts=args.accounts, n_days=args.days, seed=args.seed)
    txns, accounts = generate_book(config)
    print(f"[aml] generated {len(txns):,} transactions over {len(accounts):,} accounts; "
          f"{int(accounts['is_suspicious'].sum())} suspicious "
          f"({accounts['is_suspicious'].mean() * 100:.2f}% base rate)")

    thresholds = Thresholds()
    costs = CostModel(
        minutes_per_alert=args.minutes_per_alert,
        cost_per_missed_case=args.cost_per_missed_case,
    )

    alerts = run_all_rules(txns, thresholds, SANCTIONED_COUNTERPARTIES)
    alerts.to_csv(os.path.join(args.out, "alerts.csv"), index=False)
    database_path = write_analytics_database(
        txns,
        accounts,
        alerts,
        os.path.join(args.out, "monitoring.sqlite"),
    )

    rules_table = per_rule_table(alerts, accounts)
    summary = portfolio_summary(alerts, accounts, costs)
    missed = missed_case_analysis(alerts, accounts)

    print(f"[aml] {summary['accounts_reviewed']:,} accounts alerted; "
          f"precision {summary['precision'] * 100:.1f}%, "
          f"recall {summary['recall'] * 100:.1f}%, "
          f"{summary['alerts_per_true_case']:.1f} alerts per true case")

    sweep = threshold_sweep(
        txns,
        accounts,
        args.sweep,
        SWEEP_VALUES[args.sweep],
        base=thresholds,
        sanctioned=SANCTIONED_COUNTERPARTIES,
        costs=costs,
    )
    sweep.to_csv(os.path.join(args.out, f"sweep_{args.sweep}.csv"), index=False)
    png = cost_curve_plot(sweep, args.sweep, os.path.join(args.out, f"sweep_{args.sweep}.png"))

    cheapest = sweep.loc[sweep["total_expected_cost_eur"].idxmin()]

    lines = [
        "# Transaction monitoring - rule performance and threshold tuning",
        "",
        f"Generated {date.today().isoformat()} by `python -m aml.run_pipeline "
        f"--sweep {args.sweep}`.",
        "",
        f"Synthetic book: {len(accounts):,} accounts, {len(txns):,} transactions, "
        f"{args.days} business days, seed {args.seed}. Base rate of suspicious "
        f"accounts {accounts['is_suspicious'].mean() * 100:.2f}%.",
        "",
        "Cost assumptions, all arguable and all adjustable on the command line: "
        f"{costs.minutes_per_alert:.0f} minutes to triage one alert at "
        f"EUR {costs.analyst_cost_per_hour:.0f}/hour (EUR {costs.cost_per_alert:.2f} "
        f"per alert); EUR {costs.cost_per_missed_case:,.0f} for a missed reportable "
        f"case; EUR {costs.cost_per_missed_sanctions_case:,.0f} for a missed "
        "sanctions hit, because that one is strict liability.",
        "",
        "## Per-rule performance",
        "",
        "Recall (all) scores each rule against every suspicious account, including "
        "typologies it was never written to catch. Recall (own typology) is the "
        "number that means something.",
        "",
        to_markdown(rules_table, PER_RULE_COLS),
        "",
        "## The queue an analyst actually faces",
        "",
        "De-duplicated to accounts, so a customer caught by three rules counts once.",
        "",
        summary_markdown(summary),
        "",
        "## What the rule set missed",
        "",
        to_markdown(missed, ["typology", "population", "detected", "missed", "detection_rate"]),
        "",
        f"## Threshold sweep on `{args.sweep}`",
        "",
        "Every row is a full re-run of the rule set. Tightening a threshold always "
        "buys fewer alerts and always buys fewer detections; the only question is "
        "the exchange rate between analyst hours and missed reports.",
        "",
        to_markdown(sweep, [args.sweep] + SWEEP_COLS),
        "",
        f"![threshold sweep](sweep_{args.sweep}.png)",
        "",
        "### Reading the sweep",
        "",
        f"On these cost assumptions the cheapest setting of `{args.sweep}` is "
        f"**{cheapest[args.sweep]}**, at EUR {cheapest['total_expected_cost_eur']:,.0f} "
        f"total expected cost: {int(cheapest['accounts_reviewed']):,} accounts "
        f"reviewed, {int(cheapest['fn'])} suspicious accounts missed, "
        f"{int(cheapest['missed_sanctions_cases'])} of them sanctions cases. "
        "That is a defensible recommendation only because the assumptions behind "
        "it are written down above - change the price of a missed case and the "
        "recommendation moves, which is exactly the argument to have with a "
        "compliance officer rather than hide inside a model.",
        "",
        "### The limitation that matters most",
        "",
        "Recall here is measurable only because the data is synthetic and the "
        "typologies were injected deliberately. On a real book nobody knows the "
        "denominator: you can measure the cases you found and never the ones you "
        "did not. Every recall figure above should be read as *recall against "
        "known injected typologies*, which is an upper bound on what the same "
        "rules would achieve against an adversary who is adapting to them.",
        "",
    ]

    report_path = os.path.join(args.out, "monitoring_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    dashboard_path = os.path.join(args.out, "monitoring_dashboard.html")
    build_dashboard_html(summary, rules_table, missed, sweep, args.sweep, dashboard_path)

    print(f"[aml] wrote {report_path}")
    print(f"[aml] wrote {dashboard_path}")
    print(f"[aml] wrote {png}")
    print(f"[aml] wrote {database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
