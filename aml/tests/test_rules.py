"""Each rule, against a book small enough to check by eye.

A monitoring rule that has never been tested on a hand-built case is an opinion.
These are the cases that decide whether the rule says what its docstring claims.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aml.rules import (
    Thresholds,
    rule_dormant_reactivation,
    rule_large_cash,
    rule_pass_through,
    rule_sanctions_screening,
    rule_structuring,
    run_all_rules,
)

T = Thresholds()


def book(rows: list[tuple]) -> pd.DataFrame:
    """Build a transaction frame from ``(day_offset, account, amount, channel, cp)``."""
    base = pd.Timestamp("2026-01-05")
    return pd.DataFrame(
        [
            {
                "txn_id": f"T{i:04d}",
                "date": base + pd.Timedelta(days=day),
                "account_id": account,
                "amount": amount,
                "channel": channel,
                "counterparty": counterparty,
            }
            for i, (day, account, amount, channel, counterparty) in enumerate(rows)
        ]
    )


# --------------------------------------------------------------------------- #
# Rule 1 - large cash
# --------------------------------------------------------------------------- #
def test_large_cash_fires_on_a_cash_credit_over_the_threshold():
    txns = book([(0, "A", 12_000.0, "cash", "CP1")])
    alerts = rule_large_cash(txns, T)
    assert list(alerts["account_id"]) == ["A"]
    assert alerts["severity"].iloc[0] == "medium"


def test_large_cash_ignores_a_transfer_of_the_same_size():
    txns = book([(0, "A", 12_000.0, "transfer", "CP1")])
    assert rule_large_cash(txns, T).empty


def test_large_cash_ignores_a_large_withdrawal():
    """A debit is not a cash deposit. Sign errors here would double the alert
    volume and nobody would notice from the headline numbers."""
    txns = book([(0, "A", -12_000.0, "cash", "CP1")])
    assert rule_large_cash(txns, T).empty


def test_large_cash_is_inclusive_at_the_threshold():
    assert not rule_large_cash(book([(0, "A", 10_000.0, "cash", "CP1")]), T).empty
    assert rule_large_cash(book([(0, "A", 9_999.99, "cash", "CP1")]), T).empty


# --------------------------------------------------------------------------- #
# Rule 2 - structuring
# --------------------------------------------------------------------------- #
def test_structuring_fires_on_three_deposits_just_under_the_threshold():
    txns = book(
        [
            (0, "A", 9_200.0, "cash", "CP1"),
            (2, "A", 8_900.0, "cash", "CP2"),
            (4, "A", 9_400.0, "cash", "CP3"),
        ]
    )
    alerts = rule_structuring(txns, T)
    assert list(alerts["account_id"]) == ["A"]
    assert alerts["severity"].iloc[0] == "high"


def test_structuring_does_not_fire_below_the_minimum_count():
    txns = book([(0, "A", 9_200.0, "cash", "CP1"), (1, "A", 9_300.0, "cash", "CP2")])
    assert rule_structuring(txns, T).empty


def test_structuring_does_not_fire_when_the_deposits_are_spread_out():
    """Same three deposits, five weeks apart instead of five days. The window is
    the whole point of the rule."""
    txns = book(
        [
            (0, "A", 9_200.0, "cash", "CP1"),
            (17, "A", 8_900.0, "cash", "CP2"),
            (35, "A", 9_400.0, "cash", "CP3"),
        ]
    )
    assert rule_structuring(txns, T).empty


def test_structuring_ignores_amounts_over_the_threshold():
    """Deposits above the threshold are rule 1's business. If they also counted
    here, the two rules would be measuring the same thing twice."""
    txns = book(
        [
            (0, "A", 11_000.0, "cash", "CP1"),
            (1, "A", 12_000.0, "cash", "CP2"),
            (2, "A", 13_000.0, "cash", "CP3"),
        ]
    )
    assert rule_structuring(txns, T).empty


def test_structuring_counts_per_account_not_across_the_book():
    txns = book(
        [
            (0, "A", 9_200.0, "cash", "CP1"),
            (1, "B", 9_300.0, "cash", "CP2"),
            (2, "C", 9_400.0, "cash", "CP3"),
        ]
    )
    assert rule_structuring(txns, T).empty


# --------------------------------------------------------------------------- #
# Rule 3 - pass-through
# --------------------------------------------------------------------------- #
def test_pass_through_fires_when_money_leaves_almost_as_fast_as_it_arrived():
    txns = book(
        [
            (0, "A", 50_000.0, "transfer", "CP1"),
            (1, "A", -24_000.0, "transfer", "CP2"),
            (2, "A", -23_000.0, "transfer", "CP3"),
        ]
    )
    alerts = rule_pass_through(txns, T)
    assert list(alerts["account_id"]) == ["A"]


def test_pass_through_does_not_fire_when_the_money_stays():
    txns = book([(0, "A", 50_000.0, "transfer", "CP1")])
    assert rule_pass_through(txns, T).empty


def test_pass_through_ignores_small_inflows():
    """Ratio alone is not enough: a EUR 200 payment in and out is a current
    account working normally."""
    txns = book(
        [
            (0, "A", 200.0, "transfer", "CP1"),
            (1, "A", -195.0, "transfer", "CP2"),
        ]
    )
    assert rule_pass_through(txns, T).empty


# --------------------------------------------------------------------------- #
# Rule 4 - dormant reactivation
# --------------------------------------------------------------------------- #
def test_dormant_reactivation_fires_after_a_long_gap_and_real_volume():
    txns = book(
        [
            (0, "A", 500.0, "card", "CP1"),
            (95, "A", 20_000.0, "transfer", "CP2"),
            (96, "A", 12_000.0, "transfer", "CP3"),
        ]
    )
    alerts = rule_dormant_reactivation(txns, T)
    assert list(alerts["account_id"]) == ["A"]


def test_dormant_reactivation_needs_the_volume_too():
    txns = book([(0, "A", 500.0, "card", "CP1"), (95, "A", 300.0, "card", "CP2")])
    assert rule_dormant_reactivation(txns, T).empty


def test_dormant_reactivation_needs_the_gap_too():
    txns = book([(0, "A", 500.0, "card", "CP1"), (3, "A", 40_000.0, "transfer", "CP2")])
    assert rule_dormant_reactivation(txns, T).empty


# --------------------------------------------------------------------------- #
# Rule 5 - sanctions screening
# --------------------------------------------------------------------------- #
LIST = ("ACME TRADING FZE", "NORTHSTAR SHIPPING OOO")


def test_sanctions_screening_matches_after_normalisation():
    """Punctuation, case and a corporate suffix must not defeat the match."""
    txns = book([(0, "A", -5_000.0, "transfer", "Acme Trading F.Z.E. Ltd.")])
    alerts = rule_sanctions_screening(txns, T, LIST)
    assert list(alerts["account_id"]) == ["A"]
    assert alerts["severity"].iloc[0] == "critical"


def test_sanctions_screening_does_not_match_an_unrelated_name():
    txns = book([(0, "A", -5_000.0, "transfer", "Acme Bakeries Limited")])
    assert rule_sanctions_screening(txns, T, LIST).empty


def test_sanctions_screening_fires_on_any_amount_by_default():
    txns = book([(0, "A", -1.0, "transfer", "NORTHSTAR SHIPPING OOO")])
    assert not rule_sanctions_screening(txns, T, LIST).empty


def test_sanctions_screening_with_an_empty_list_fires_nothing():
    txns = book([(0, "A", -5_000.0, "transfer", "ACME TRADING FZE")])
    assert rule_sanctions_screening(txns, T, ()).empty


# --------------------------------------------------------------------------- #
# The whole set
# --------------------------------------------------------------------------- #
def test_run_all_rules_returns_one_row_per_rule_per_account():
    """An account can be alerted by more than one rule, and the frame keeps both
    rows. De-duplication belongs in the evaluation, not here."""
    txns = book(
        [
            (0, "A", 9_500.0, "cash", "CP1"),
            (1, "A", 9_300.0, "cash", "CP2"),
            (2, "A", 9_100.0, "cash", "CP3"),
            (3, "A", 12_000.0, "cash", "CP4"),
        ]
    )
    alerts = run_all_rules(txns, T, LIST)
    assert set(alerts["rule"]) == {"large_cash", "structuring"}
    assert set(alerts["account_id"]) == {"A"}


def test_run_all_rules_on_an_empty_book_is_empty_not_an_error():
    empty = pd.DataFrame(
        columns=["txn_id", "date", "account_id", "amount", "channel", "counterparty"]
    )
    empty["date"] = pd.to_datetime(empty["date"])
    empty["amount"] = empty["amount"].astype(float)
    alerts = run_all_rules(empty, T, LIST)
    assert alerts.empty


def test_thresholds_are_immutable():
    """The sweep relies on dataclasses.replace rather than mutation, so that a
    run cannot quietly change the settings of the run before it."""
    with pytest.raises(Exception):
        T.large_cash = 1.0  # type: ignore[misc]
