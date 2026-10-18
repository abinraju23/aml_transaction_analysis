"""The measurement, checked against arithmetic done on paper.

If the cost model is wrong, the recommendation at the bottom of the report is
wrong, and a recommendation with a euro figure attached is the part a reader will
quote back. So it gets tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aml.data import SANCTIONED_COUNTERPARTIES, BookConfig, generate_book
from aml.evaluate import (
    CostModel,
    confusion,
    missed_case_analysis,
    per_rule_table,
    portfolio_summary,
    threshold_sweep,
)
from aml.rules import Thresholds, run_all_rules


def accounts_frame(spec: dict[str, str]) -> pd.DataFrame:
    """``{account_id: typology}``, with ``none`` meaning a clean customer."""
    return pd.DataFrame(
        {
            "account_id": list(spec),
            "opened": pd.Timestamp("2024-01-01"),
            "segment": "retail",
            "is_pep": False,
            "typology": list(spec.values()),
            "is_suspicious": [t != "none" for t in spec.values()],
        }
    )


def alerts_frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account_id": account,
                "rule": rule,
                "first_date": pd.Timestamp("2026-01-05"),
                "detail": "",
                "severity": "medium",
            }
            for account, rule in pairs
        ],
        columns=["account_id", "rule", "first_date", "detail", "severity"],
    )


# --------------------------------------------------------------------------- #
# Confusion counting
# --------------------------------------------------------------------------- #
def test_confusion_counts_are_account_level():
    accounts = accounts_frame(
        {"A": "structuring", "B": "none", "C": "sanctions", "D": "none"}
    )
    # Alerted A (true), B (false). Missed C. Left D alone (true negative).
    counts = confusion(["A", "B"], accounts)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_confusion_ignores_duplicate_alerts_on_the_same_account():
    accounts = accounts_frame({"A": "structuring", "B": "none"})
    assert confusion(["A", "A", "A"], accounts) == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}


def test_alerting_nobody_scores_zero_recall_not_an_error():
    accounts = accounts_frame({"A": "structuring", "B": "none"})
    summary = portfolio_summary(alerts_frame([]), accounts)
    assert summary["recall"] == pytest.approx(0.0)
    assert np.isnan(summary["precision"])  # no alerts means precision undefined
    assert summary["alerts_per_true_case"] == float("inf")


# --------------------------------------------------------------------------- #
# Rates
# --------------------------------------------------------------------------- #
def test_precision_and_recall_match_the_hand_calculation():
    accounts = accounts_frame(
        {
            "A": "structuring",
            "B": "structuring",
            "C": "pass_through",
            "D": "none",
            "E": "none",
            "F": "none",
        }
    )
    # Three alerts: two real (A, B), one false (D). C is missed.
    summary = portfolio_summary(alerts_frame([("A", "structuring"), ("B", "structuring"), ("D", "large_cash")]), accounts)

    assert summary["tp"] == 2 and summary["fp"] == 1 and summary["fn"] == 1
    assert summary["precision"] == pytest.approx(2 / 3)
    assert summary["recall"] == pytest.approx(2 / 3)
    assert summary["false_positive_rate"] == pytest.approx(1 / 3)  # 1 FP of 3 clean
    assert summary["alerts_per_true_case"] == pytest.approx(1.5)


def test_duplicate_rows_are_reported_but_not_double_counted():
    accounts = accounts_frame({"A": "structuring", "B": "none"})
    alerts = alerts_frame([("A", "structuring"), ("A", "large_cash")])
    summary = portfolio_summary(alerts, accounts)

    assert summary["raw_alert_rows"] == 2
    assert summary["accounts_reviewed"] == 1, "the analyst sees one customer, not two"
    assert summary["duplicate_alert_rows"] == 1


def test_per_rule_table_scores_each_rule_against_its_own_typology():
    accounts = accounts_frame(
        {"A": "structuring", "B": "sanctions", "C": "none", "D": "pass_through"}
    )
    alerts = alerts_frame([("A", "structuring"), ("C", "structuring")])
    table = per_rule_table(alerts, accounts).set_index("rule")

    row = table.loc["structuring"]
    assert row["tp"] == 1 and row["fp"] == 1
    assert row["precision"] == pytest.approx(0.5)
    # Two of the three suspicious accounts are other typologies, so recall
    # against the whole population is 1/3 while recall against its own target
    # population is 1/1.
    assert row["recall"] == pytest.approx(1 / 3)
    assert row["target_recall"] == pytest.approx(1.0)
    assert table.loc["sanctions_screening"]["alerts"] == 0


# --------------------------------------------------------------------------- #
# The cost model
# --------------------------------------------------------------------------- #
def test_cost_per_alert_is_minutes_times_rate():
    costs = CostModel(minutes_per_alert=20.0, analyst_cost_per_hour=30.0)
    assert costs.cost_per_alert == pytest.approx(10.0)


def test_total_expected_cost_is_triage_plus_missed_cases():
    accounts = accounts_frame({"A": "structuring", "B": "none", "C": "pass_through"})
    alerts = alerts_frame([("A", "structuring"), ("B", "large_cash")])
    costs = CostModel(
        minutes_per_alert=30.0,
        analyst_cost_per_hour=40.0,          # EUR 20 an alert
        cost_per_missed_case=10_000.0,
        cost_per_missed_sanctions_case=100_000.0,
    )
    summary = portfolio_summary(alerts, accounts, costs)

    assert summary["analyst_hours"] == pytest.approx(1.0)      # 2 alerts x 30 min
    assert summary["triage_cost_eur"] == pytest.approx(40.0)   # 2 x EUR 20
    assert summary["missed_case_cost_eur"] == pytest.approx(10_000.0)  # C missed
    assert summary["total_expected_cost_eur"] == pytest.approx(10_040.0)


def test_a_missed_sanctions_case_is_priced_far_above_the_others():
    accounts = accounts_frame({"A": "sanctions", "B": "structuring"})
    costs = CostModel(cost_per_missed_case=10_000.0, cost_per_missed_sanctions_case=100_000.0)
    summary = portfolio_summary(alerts_frame([]), accounts, costs)

    assert summary["missed_sanctions_cases"] == 1
    assert summary["missed_case_cost_eur"] == pytest.approx(110_000.0)


# --------------------------------------------------------------------------- #
# The sweep, on a real generated book
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_book():
    return generate_book(BookConfig(n_accounts=300, n_days=90, seed=7))


def test_generated_book_has_a_low_base_rate_and_every_typology(small_book):
    _, accounts = small_book
    assert 0.0 < accounts["is_suspicious"].mean() < 0.10, "a realistic base rate is small"
    assert set(accounts["typology"]) == {
        "none",
        "structuring",
        "pass_through",
        "dormant",
        "sanctions",
    }


def test_the_book_is_deterministic_given_a_seed():
    first, _ = generate_book(BookConfig(n_accounts=120, n_days=40, seed=99))
    second, _ = generate_book(BookConfig(n_accounts=120, n_days=40, seed=99))
    pd.testing.assert_frame_equal(first, second)


def test_raising_a_threshold_cannot_increase_the_alert_count(small_book):
    """The monotonicity that makes the sweep interpretable. If this fails, the
    tuning table cannot be read as a trade-off at all."""
    txns, accounts = small_book
    sweep = threshold_sweep(
        txns,
        accounts,
        "large_cash",
        [5_000, 10_000, 25_000, 100_000],
        sanctioned=SANCTIONED_COUNTERPARTIES,
    )
    volumes = list(sweep["accounts_reviewed"])
    assert volumes == sorted(volumes, reverse=True)


def test_missed_case_analysis_adds_up(small_book):
    txns, accounts = small_book
    alerts = run_all_rules(txns, Thresholds(), SANCTIONED_COUNTERPARTIES)
    missed = missed_case_analysis(alerts, accounts)

    assert (missed["detected"] + missed["missed"] == missed["population"]).all()
    assert missed["population"].sum() == int(accounts["is_suspicious"].sum())


def test_sanctions_rule_catches_every_sanctions_account_in_the_book(small_book):
    """The one rule in the set that should have perfect recall against its own
    typology, because the generator puts a listed counterparty on every one of
    those accounts and the match is exact. If this drops below 1.0, the name
    normalisation has broken."""
    txns, accounts = small_book
    alerts = run_all_rules(txns, Thresholds(), SANCTIONED_COUNTERPARTIES)
    table = per_rule_table(alerts, accounts).set_index("rule")
    assert table.loc["sanctions_screening"]["target_recall"] == pytest.approx(1.0)
