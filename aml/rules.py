"""Transaction-monitoring rules.

Five rules, each with explicit thresholds, each returning account-level alerts
with the reason attached. Thresholds are arguments rather than constants because
the deliverable of this project is the **threshold-tuning decision**, not the
rules themselves: every one of these numbers trades analyst hours against missed
reports, and that trade is the thing a regulator asks you to defend.

Vocabulary, so the write-up matches how the industry says it: a rule fires an
**alert**; an analyst triages alerts into **cases**; a case that survives review
becomes a **SAR** (in Ireland, a suspicious transaction report to the FIU and
Revenue). This module models the first step only. Nothing here decides anything
about a real customer.

Every rule returns the same shape, so the evaluation code does not special-case
any of them:

    account_id | rule | first_date | detail | severity
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

ALERT_COLUMNS = ["account_id", "rule", "first_date", "detail", "severity"]


@dataclass(frozen=True)
class Thresholds:
    """Every tunable number in one place, so a sweep is a loop over this object.

    Defaults are a plausible starting point, not a recommendation. The point of
    ``evaluate.threshold_sweep`` is that the defaults are almost certainly wrong
    and the cost of being wrong is measurable.
    """

    # Rule 1 - large cash
    large_cash: float = 10_000.0
    # Rule 2 - structuring
    structuring_band_low: float = 0.80   # fraction of the reporting threshold
    structuring_band_high: float = 1.00
    structuring_window_days: int = 7
    structuring_min_count: int = 3
    # Rule 3 - pass-through
    pass_through_window_days: int = 5
    pass_through_ratio: float = 0.85     # outflow / inflow inside the window
    pass_through_min_inflow: float = 20_000.0
    # Rule 4 - dormant reactivation
    dormant_days: int = 60
    dormant_burst_amount: float = 25_000.0
    dormant_burst_window_days: int = 10
    # Rule 5 - sanctions screening
    sanctions_min_amount: float = 0.0    # screen every payment by default


def _empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=ALERT_COLUMNS)


def _alerts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=ALERT_COLUMNS) if rows else _empty_alerts()


# --------------------------------------------------------------------------- #
# Rule 1 - a single large cash credit
# --------------------------------------------------------------------------- #
def rule_large_cash(txns: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """Any single cash credit at or above the reporting threshold.

    The crudest rule in any monitoring system, and the one that generates the
    most false positives, because plenty of legitimate businesses bank large
    cash. It is kept precisely so the write-up can show that.
    """
    hits = txns[
        (txns["channel"] == "cash")
        & (txns["amount"] >= t.large_cash)
    ]
    rows = [
        {
            "account_id": account_id,
            "rule": "large_cash",
            "first_date": group["date"].min(),
            "detail": f"{len(group)} cash credit(s) >= {t.large_cash:,.0f}; "
                      f"largest {group['amount'].max():,.0f}",
            "severity": "medium",
        }
        for account_id, group in hits.groupby("account_id")
    ]
    return _alerts(rows)


# --------------------------------------------------------------------------- #
# Rule 2 - structuring (smurfing)
# --------------------------------------------------------------------------- #
def rule_structuring(txns: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """Several cash credits *just under* the threshold inside a short window.

    This is the rule that catches what rule 1 is designed to miss, and the
    interesting parameter is the band: too narrow and a launderer one euro lower
    walks through, too wide and every ordinary four-figure deposit alerts.
    """
    band_low = t.structuring_band_low * t.large_cash
    band_high = t.structuring_band_high * t.large_cash
    in_band = txns[
        (txns["channel"] == "cash")
        & (txns["amount"] >= band_low)
        & (txns["amount"] < band_high)
    ]
    if in_band.empty:
        return _empty_alerts()

    window = f"{t.structuring_window_days}D"
    rows = []
    for account_id, group in in_band.groupby("account_id"):
        series = (
            group.set_index("date")["amount"].sort_index()
        )
        counts = series.rolling(window).count()
        if (counts >= t.structuring_min_count).any():
            fired = counts[counts >= t.structuring_min_count]
            rows.append(
                {
                    "account_id": account_id,
                    "rule": "structuring",
                    "first_date": fired.index.min(),
                    "detail": f"{int(counts.max())} cash credits in "
                              f"[{band_low:,.0f}, {band_high:,.0f}) within "
                              f"{t.structuring_window_days}d",
                    "severity": "high",
                }
            )
    return _alerts(rows)


# --------------------------------------------------------------------------- #
# Rule 3 - pass-through / funnel account
# --------------------------------------------------------------------------- #
def rule_pass_through(txns: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """Money in, nearly all of it straight back out, inside a few days.

    Measured as outflow over inflow within a rolling window, which is how a
    funnel account looks from the bank's side: high turnover, negligible
    balance, no economic purpose on the face of it.
    """
    window = f"{t.pass_through_window_days}D"
    rows = []
    for account_id, group in txns.groupby("account_id"):
        g = group.sort_values("date").set_index("date")
        inflow = g["amount"].clip(lower=0).rolling(window).sum()
        outflow = (-g["amount"]).clip(lower=0).rolling(window).sum()
        eligible = inflow >= t.pass_through_min_inflow
        if not eligible.any():
            continue
        ratio = (outflow / inflow.where(inflow > 0)).where(eligible)
        fired = ratio[ratio >= t.pass_through_ratio].dropna()
        if not fired.empty:
            rows.append(
                {
                    "account_id": account_id,
                    "rule": "pass_through",
                    "first_date": fired.index.min(),
                    "detail": f"outflow/inflow {fired.max():.2f} over "
                              f"{t.pass_through_window_days}d on inflow >= "
                              f"{t.pass_through_min_inflow:,.0f}",
                    "severity": "high",
                }
            )
    return _alerts(rows)


# --------------------------------------------------------------------------- #
# Rule 4 - dormant account reactivation
# --------------------------------------------------------------------------- #
def rule_dormant_reactivation(txns: pd.DataFrame, t: Thresholds) -> pd.DataFrame:
    """An account quiet for months that suddenly moves real volume.

    A classic mule indicator, and a rule with an obvious seasonal false
    positive: an account used once a year for a tax bill looks identical.
    """
    rows = []
    for account_id, group in txns.groupby("account_id"):
        g = group.sort_values("date")
        gaps = g["date"].diff().dt.days
        reactivations = g.loc[gaps >= t.dormant_days, "date"]
        for when in reactivations:
            burst_end = when + pd.Timedelta(days=t.dormant_burst_window_days)
            burst = g[(g["date"] >= when) & (g["date"] <= burst_end)]
            volume = float(burst["amount"].abs().sum())
            if volume >= t.dormant_burst_amount:
                rows.append(
                    {
                        "account_id": account_id,
                        "rule": "dormant_reactivation",
                        "first_date": when,
                        "detail": f"{t.dormant_days}d+ inactive, then "
                                  f"{volume:,.0f} in "
                                  f"{t.dormant_burst_window_days}d",
                        "severity": "medium",
                    }
                )
                break  # one alert per account is enough for triage
    return _alerts(rows)


# --------------------------------------------------------------------------- #
# Rule 5 - sanctions / PEP screening
# --------------------------------------------------------------------------- #
def _normalise_name(name: str) -> str:
    """Screening names the way a screening engine does: upper case, no
    punctuation, no corporate suffixes, single spaces.

    The important detail is that abbreviations such as ``F.Z.E.`` and ``S.A.`` are
    the same token as their condensed form ``FZE`` and ``SA``. Otherwise a name can
    be incorrectly normalised into ``F Z E`` and fail an exact match.
    """
    cleaned = re.sub(r"[^A-Z0-9\s]", " ", str(name).upper())
    tokens = cleaned.split()

    merged: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if len(token) == 1 and i + 1 < len(tokens):
            letters = [token]
            j = i
            while j + 1 < len(tokens) and len(tokens[j + 1]) == 1 and re.fullmatch(r"[A-Z]", tokens[j + 1]):
                letters.append(tokens[j + 1])
                j += 1
            if len(letters) > 1:
                merged.append("".join(letters))
                i = j + 1
                continue
        merged.append(token)
        i += 1

    stop = {"LTD", "LIMITED", "SA", "OOO", "FZE", "GMBH", "PLC", "LLC", "INC", "BV"}
    return " ".join(tok for tok in merged if tok not in stop)


def rule_sanctions_screening(
    txns: pd.DataFrame,
    t: Thresholds,
    sanctioned: tuple[str, ...],
) -> pd.DataFrame:
    """Exact match against a screening list, after name normalisation.

    Screening is the one rule here where a miss is not a judgement call: paying
    a listed party is a breach whatever the amount, which is why
    ``sanctions_min_amount`` defaults to zero and why the cost model in
    :mod:`aml.evaluate` prices a miss on this rule far above the others.
    """
    if txns.empty:
        return _empty_alerts()
    watchlist = {_normalise_name(name) for name in sanctioned}
    screened = txns.assign(_norm=txns["counterparty"].map(_normalise_name))
    hits = screened[
        screened["_norm"].isin(watchlist)
        & (screened["amount"].abs() >= t.sanctions_min_amount)
    ]
    rows = [
        {
            "account_id": account_id,
            "rule": "sanctions_screening",
            "first_date": group["date"].min(),
            "detail": f"payment(s) to listed party: "
                      f"{sorted(set(group['counterparty']))[0]}",
            "severity": "critical",
        }
        for account_id, group in hits.groupby("account_id")
    ]
    return _alerts(rows)


# --------------------------------------------------------------------------- #
# Running the whole rule set
# --------------------------------------------------------------------------- #
def run_all_rules(
    txns: pd.DataFrame,
    thresholds: Thresholds | None = None,
    sanctioned: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Every rule over the same book, concatenated.

    One account can be alerted by several rules, which is realistic and is why
    :mod:`aml.evaluate` reports both per-rule and de-duplicated account-level
    numbers. Quoting only the per-rule figures hides how much of the alert
    volume is the same customer arriving three times.
    """
    t = thresholds or Thresholds()
    frames = [
        rule_large_cash(txns, t),
        rule_structuring(txns, t),
        rule_pass_through(txns, t),
        rule_dormant_reactivation(txns, t),
        rule_sanctions_screening(txns, t, sanctioned),
    ]
    out = pd.concat(frames, ignore_index=True) if frames else _empty_alerts()
    return out.reset_index(drop=True)
