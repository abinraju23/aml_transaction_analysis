"""The deliverable: what the rules cost, and what they miss.

A classifier is not the point of this project. The point is the measurement, and
in particular the asymmetry that makes AML monitoring different from an ordinary
classification problem:

* a **false positive** costs analyst hours - real money, paid every single time;
* a **false negative** is a potentially reportable breach - rare, but priced in
  regulatory exposure, remediation and reputational damage.

Those two are not comparable in units, so a single accuracy number is not just
uninformative, it is misleading. Everything here is reported in alerts, hours
and euro, with the exchange rate between them stated as an explicit assumption
that a reader can disagree with.

Recall is measured against the injected ground truth, which only exists because
the data is synthetic. On a real book nobody knows the denominator - you can
only measure the cases you found - and that limitation is the single most
important sentence in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from .rules import Thresholds, run_all_rules

# Which typology each rule is actually written to catch. Used to report a rule
# against its own target as well as against the whole suspicious population,
# because judging a sanctions rule on its structuring recall is meaningless.
RULE_TARGETS = {
    "large_cash": {"structuring", "dormant"},
    "structuring": {"structuring"},
    "pass_through": {"pass_through"},
    "dormant_reactivation": {"dormant"},
    "sanctions_screening": {"sanctions"},
}


@dataclass(frozen=True)
class CostModel:
    """The exchange rate between analyst time and regulatory risk.

    Every number here is an assumption, and the sweep in
    :func:`threshold_sweep` is worth more than any single setting of them. The
    defaults are deliberately conservative and sourced from how these teams
    actually staff: a triage analyst on roughly EUR 45k fully loaded, working
    through alerts at twenty minutes each.
    """

    minutes_per_alert: float = 20.0
    analyst_cost_per_hour: float = 32.0
    # What a missed case is priced at. Not a prediction of a fine - it is the
    # expected cost of remediation, lookback and exposure for one missed
    # reportable case. Argue with it by changing it.
    cost_per_missed_case: float = 25_000.0
    # A missed sanctions hit is categorically worse: strict liability, and the
    # breach is the payment itself rather than a judgement about suspicion.
    cost_per_missed_sanctions_case: float = 250_000.0

    @property
    def cost_per_alert(self) -> float:
        return self.minutes_per_alert / 60.0 * self.analyst_cost_per_hour


def confusion(
    alerted_accounts: Iterable[str],
    accounts: pd.DataFrame,
) -> dict:
    """Account-level confusion counts against the injected labels."""
    alerted = set(alerted_accounts)
    suspicious = set(accounts.loc[accounts["is_suspicious"], "account_id"])
    all_ids = set(accounts["account_id"])

    tp = len(alerted & suspicious)
    fp = len(alerted - suspicious)
    fn = len(suspicious - alerted)
    tn = len(all_ids - alerted - suspicious)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rates(counts: dict) -> dict:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )
    return {
        **counts,
        "alerts": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
        # The number an operations manager actually asks for: how many alerts an
        # analyst works through to find one real case.
        "alerts_per_true_case": (tp + fp) / tp if tp else float("inf"),
    }


def per_rule_table(alerts: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """One row per rule: volume, precision, recall against all suspicious
    accounts, and recall against the typology the rule targets."""
    rows = []
    labels = accounts.set_index("account_id")["typology"]
    for rule in RULE_TARGETS:
        hit = alerts.loc[alerts["rule"] == rule, "account_id"].unique()
        counts = confusion(hit, accounts)
        row = {"rule": rule, **_rates(counts)}

        targets = RULE_TARGETS[rule]
        target_ids = set(labels[labels.isin(targets)].index)
        row["target_typologies"] = "|".join(sorted(targets))
        row["target_population"] = len(target_ids)
        row["target_recall"] = (
            len(set(hit) & target_ids) / len(target_ids) if target_ids else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_summary(
    alerts: pd.DataFrame,
    accounts: pd.DataFrame,
    costs: CostModel | None = None,
) -> dict:
    """The whole rule set, de-duplicated to accounts, priced.

    De-duplication matters: the per-rule rows double-count any customer that
    several rules alert on, so the per-rule precision figures look better than
    the queue an analyst actually faces.
    """
    cm = costs or CostModel()
    unique_alerted = alerts["account_id"].unique() if len(alerts) else []
    counts = confusion(unique_alerted, accounts)
    rates = _rates(counts)

    labels = accounts.set_index("account_id")["typology"]
    missed = set(accounts.loc[accounts["is_suspicious"], "account_id"]) - set(unique_alerted)
    missed_sanctions = sum(1 for account in missed if labels.get(account) == "sanctions")
    missed_other = len(missed) - missed_sanctions

    fp_cost = rates["fp"] * cm.cost_per_alert
    tp_cost = rates["tp"] * cm.cost_per_alert  # true cases cost time to work too
    fn_cost = (
        missed_other * cm.cost_per_missed_case
        + missed_sanctions * cm.cost_per_missed_sanctions_case
    )

    return {
        **rates,
        "raw_alert_rows": int(len(alerts)),
        "duplicate_alert_rows": int(len(alerts) - len(set(unique_alerted))),
        "accounts_reviewed": int(rates["alerts"]),
        "analyst_hours": rates["alerts"] * cm.minutes_per_alert / 60.0,
        "triage_cost_eur": fp_cost + tp_cost,
        "false_positive_cost_eur": fp_cost,
        "missed_case_cost_eur": fn_cost,
        "missed_sanctions_cases": int(missed_sanctions),
        "total_expected_cost_eur": fp_cost + tp_cost + fn_cost,
    }


def threshold_sweep(
    txns: pd.DataFrame,
    accounts: pd.DataFrame,
    field: str,
    values: Iterable,
    base: Thresholds | None = None,
    sanctioned: tuple[str, ...] = (),
    costs: CostModel | None = None,
) -> pd.DataFrame:
    """Re-run the whole rule set across values of one threshold.

    This is the table the project exists to produce. It turns "where should the
    threshold be?" from an opinion into a line you can point at, and it shows
    the shape that makes AML tuning hard: tightening a threshold always buys
    fewer alerts and always buys fewer detections, and the only question is the
    exchange rate.
    """
    base = base or Thresholds()
    rows = []
    for value in values:
        thresholds = replace(base, **{field: value})
        alerts = run_all_rules(txns, thresholds, sanctioned)
        summary = portfolio_summary(alerts, accounts, costs)
        rows.append({field: value, **summary})
    return pd.DataFrame(rows)


def missed_case_analysis(
    alerts: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """What the rule set missed, grouped by typology.

    Written up as prose in the README, because "we miss 40% of pass-through
    accounts because the ratio threshold is above where the behaviour sits" is a
    finding, and a recall number on its own is not.
    """
    alerted = set(alerts["account_id"].unique() if len(alerts) else [])
    suspicious = accounts[accounts["is_suspicious"]].copy()
    suspicious["detected"] = suspicious["account_id"].isin(alerted)
    out = (
        suspicious.groupby("typology")
        .agg(population=("account_id", "size"), detected=("detected", "sum"))
        .reset_index()
    )
    out["missed"] = out["population"] - out["detected"]
    out["detection_rate"] = out["detected"] / out["population"]
    return out.sort_values("missed", ascending=False)
