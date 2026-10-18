"""Tables and charts. Markdown written by hand so the package needs no tabulate."""

from __future__ import annotations

import html
import os
from typing import Mapping, Sequence

import pandas as pd

PERCENT = {
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "target_recall",
    "detection_rate",
}
EURO = {
    "triage_cost_eur",
    "false_positive_cost_eur",
    "missed_case_cost_eur",
    "total_expected_cost_eur",
}
HEADERS = {
    "rule": "Rule",
    "alerts": "Alerts",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
    "tn": "TN",
    "precision": "Precision",
    "recall": "Recall (all)",
    "target_recall": "Recall (own typology)",
    "target_population": "Target pop.",
    "f1": "F1",
    "false_positive_rate": "FP rate",
    "alerts_per_true_case": "Alerts per true case",
    "analyst_hours": "Analyst hours",
    "triage_cost_eur": "Triage cost",
    "false_positive_cost_eur": "FP cost",
    "missed_case_cost_eur": "Missed-case cost",
    "total_expected_cost_eur": "Total expected cost",
    "missed_sanctions_cases": "Missed sanctions",
    "accounts_reviewed": "Accounts reviewed",
    "population": "Population",
    "detected": "Detected",
    "missed": "Missed",
    "detection_rate": "Detection rate",
    "typology": "Typology",
}


def _fmt(col: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    if col in PERCENT:
        return f"{value * 100:.1f}%"
    if col in EURO:
        return f"EUR {value:,.0f}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _money(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"EUR {float(value):,.0f}"


def to_markdown(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    columns = list(columns or df.columns)
    head = "| " + " | ".join(HEADERS.get(c, c) for c in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_fmt(c, row.get(c)) for c in columns) + " |"
        for row in df.to_dict("records")
    ]
    return "\n".join([head, rule, *body])


def summary_markdown(summary: Mapping) -> str:
    """The portfolio summary as a two-column table, which reads better than a
    one-row wide table when there are twenty fields."""
    order = [
        "accounts_reviewed",
        "raw_alert_rows",
        "duplicate_alert_rows",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "false_positive_rate",
        "alerts_per_true_case",
        "missed_sanctions_cases",
        "analyst_hours",
        "false_positive_cost_eur",
        "missed_case_cost_eur",
        "total_expected_cost_eur",
    ]
    lines = ["| Measure | Value |", "| --- | --- |"]
    for key in order:
        if key in summary:
            lines.append(f"| {HEADERS.get(key, key)} | {_fmt(key, summary[key])} |")
    return "\n".join(lines)


def _html_table(df: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    columns = list(columns or df.columns)
    head = "\n".join(
        [
            "<thead><tr>",
            *[f"<th>{html.escape(HEADERS.get(c, c))}</th>" for c in columns],
            "</tr></thead>",
        ]
    )
    rows = []
    for row in df.to_dict("records"):
        cell_values = [
            f"<td>{html.escape(str(_fmt(c, row.get(c))))}</td>" for c in columns
        ]
        rows.append("<tr>" + "".join(cell_values) + "</tr>")
    return f"<table><tbody>{head}{''.join(rows)}</tbody></table>"


def _sparkline_svg(values: Sequence[float], width: int = 290, height: int = 110) -> str:
    if not values:
        return "<svg viewBox='0 0 290 110' preserveAspectRatio='none'></svg>"
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v if max_v != min_v else 1
    points = []
    for i, value in enumerate(values):
        x = (i / (len(values) - 1)) * (width - 12) + 6
        y = height - 10 - ((value - min_v) / span) * (height - 24)
        points.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(points)
    return (
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='none' "
        "role='img' aria-label='cost trend'>"
        f"<polyline points='{polyline}' fill='none' stroke='#58d0b2' stroke-width='3' />"
        "</svg>"
    )


def build_dashboard_html(
    summary: Mapping,
    rules_table: pd.DataFrame,
    missed: pd.DataFrame,
    sweep: pd.DataFrame,
    field: str,
    out_path: str,
) -> str:
    """A readable HTML dashboard for the AML decision trade-off.

    The goal is to look more like a custom product than a generic notebook dump,
    while keeping the actual logic and numbers faithful to the markdown report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cards = [
        ("Accounts reviewed", summary.get("accounts_reviewed", 0)),
        ("Precision", summary.get("precision", 0.0)),
        ("Recall", summary.get("recall", 0.0)),
        ("Total cost", summary.get("total_expected_cost_eur", 0.0)),
        ("Alerts / true case", summary.get("alerts_per_true_case", 0.0)),
    ]
    def card_value(label: str, value) -> str:
        if label == "Total cost":
            return _money(value)
        if label == "Accounts reviewed":
            return f"{int(value):,}"
        if label in {"Precision", "Recall"}:
            return f"{float(value) * 100:.1f}%"
        if label == "Alerts / true case":
            return f"{float(value):,.2f}"
        return str(value)

    cards_html = "".join(
        f"<div class='card'><span class='label'>{html.escape(label)}</span><strong>{html.escape(card_value(label, value))}</strong></div>"
        for label, value in cards
    )

    recommended_threshold = None
    recommended_cost = None
    if not sweep.empty and field in sweep.columns:
        recommended = sweep.loc[sweep["total_expected_cost_eur"].idxmin()]
        recommended_threshold = recommended[field]
        recommended_cost = recommended["total_expected_cost_eur"]

    preferred_html = ""
    if recommended_threshold is not None and recommended_cost is not None:
        preferred_html = (
            "<div class='section'>"
            "<h2>Preferred threshold</h2>"
            f"<p><strong>{html.escape(str(field))}</strong> = <strong>{html.escape(str(recommended_threshold))}</strong> gives the lowest total expected cost of <strong>{html.escape(_money(recommended_cost))}</strong> under the current assumptions.</p>"
            "</div>"
        )

    rule_table_html = _html_table(rules_table, ["rule", "alerts", "precision", "recall", "target_recall", "alerts_per_true_case"])
    missed_table_html = _html_table(missed, ["typology", "population", "detected", "missed", "detection_rate"])
    sweep_table_html = _html_table(sweep, [field, "accounts_reviewed", "tp", "fp", "fn", "precision", "recall", "total_expected_cost_eur"])

    cost_values = sweep[field].tolist() if not sweep.empty else []
    cost_curve = _sparkline_svg([float(v) for v in sweep["total_expected_cost_eur"].tolist()])

    html_doc = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>AML Decision Lens</title>
  <style>
    :root {{
      --bg: #07141d;
      --panel: #0d2232;
      --panel-alt: #102b3d;
      --line: #2a465f;
      --text: #ecf5ff;
      --muted: #9bb6c9;
      --teal: #5fe3c6;
      --gold: #f4c86c;
      --brand: #7cc6ff;
      --danger: #ff8d8d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: Inter, Segoe UI, sans-serif; background: linear-gradient(180deg, #07141d 0%, #0b1a26 100%); color: var(--text);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 20px 48px; }}
    .hero {{
      display: flex; justify-content: space-between; gap: 20px; align-items: end; border: 1px solid var(--line); background: rgba(16,43,61,0.9); border-radius: 16px; padding: 24px 26px; box-shadow: 0 15px 35px rgba(0,0,0,0.18);
    }}
    .eyebrow {{ color: var(--teal); text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px; font-weight: 700; margin-bottom: 8px; }}
    h1 {{ font-size: clamp(2rem, 3vw, 3.1rem); margin: 0; }}
    .subtitle {{ color: var(--muted); max-width: 700px; line-height: 1.55; margin-top: 12px; }}
    .badge {{ background: rgba(95, 227, 198, 0.12); border: 1px solid rgba(95,227,198,0.4); color: var(--teal); border-radius: 999px; padding: 10px 14px; font-size: 12px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-top: 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px 18px 16px; }}
    .card .label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}
    .card strong {{ display: block; font-size: clamp(1.3rem, 2vw, 2rem); font-weight: 800; }}
    .card:nth-child(4) strong {{ color: var(--gold); }}
    .section {{ margin-top: 28px; background: rgba(13,34,50,0.85); border: 1px solid var(--line); border-radius: 16px; padding: 22px; }}
    .section h2 {{ margin: 0 0 14px; font-size: 1.15rem; letter-spacing: 0.03em; }}
    .section p {{ color: var(--muted); line-height: 1.6; margin-top: 0; }}
    .two-column {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: middle; }}
    th {{ background: rgba(124,198,255,0.06); color: var(--brand); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    td {{ color: var(--text); font-size: 14px; }}
    .spark {{ max-width: 320px; width: 100%; margin-top: 8px; border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: rgba(0,0,0,0.12); }}
    .mono {{ font-family: 'Cascadia Code', Consolas, monospace; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 820px) {{ .two-column {{ grid-template-columns: 1fr; }} .hero {{ flex-direction: column; align-items: flex-start; }} }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='hero'>
      <div>
        <div class='eyebrow'>AML Decision Lens</div>
        <h1>Trade-off dashboard</h1>
        <div class='subtitle'>This view focuses on the real operational question: how much analyst burden and regulatory exposure do we accept for each threshold decision?</div>
      </div>
      <div class='badge'>Threshold: {html.escape(field)}</div>
    </div>

    <div class='grid'>
      {cards_html}
    </div>

    {preferred_html}

    <div class='section'>
      <h2>Summary</h2>
      <p>The current configuration catches most known suspicious accounts but produces a high false-positive queue. The cost model makes the practical decision clear: the threshold choice is not a modelling trick; it is an operating policy.</p>
      <div class='two-column'>
        <div>
          {rule_table_html}
        </div>
        <div>
          <div class='spark'>{cost_curve}</div>
          <p class='muted'>Cost trend across the sweep. A steeper rise usually means the system is starting to miss more reportable cases than it saves in analyst time.</p>
        </div>
      </div>
    </div>

    <div class='section'>
      <h2>Missed cases by typology</h2>
      {missed_table_html}
    </div>

    <div class='section'>
      <h2>Threshold sweep</h2>
      {sweep_table_html}
    </div>
  </div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return html_doc


def cost_curve_plot(sweep: pd.DataFrame, field: str, path: str) -> str:
    """Alert volume and total expected cost against the threshold.

    Two axes on purpose: the alert count is what operations feels and the cost
    curve is what the decision turns on, and showing only one of them is how
    these decisions get made badly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)
    ax.plot(sweep[field], sweep["accounts_reviewed"], marker="o",
            linewidth=1.5, label="Accounts reviewed")
    ax.set_xlabel(field)
    ax.set_ylabel("Accounts reviewed")
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    twin.plot(sweep[field], sweep["total_expected_cost_eur"], marker="s",
              linewidth=1.5, linestyle="--", color="firebrick",
              label="Total expected cost")
    twin.set_ylabel("Total expected cost (EUR)")

    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper center")
    ax.set_title(f"Threshold sweep on {field}: volume against expected cost")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
