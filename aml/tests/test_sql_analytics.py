from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from aml.sql_analytics import write_analytics_database


@pytest.fixture
def analytics_database(tmp_path):
    transactions = pd.DataFrame(
        [
            {
                "txn_id": "T1",
                "date": pd.Timestamp("2026-01-05"),
                "account_id": "A",
                "amount": 100.0,
                "channel": "cash",
                "counterparty": "CP1",
            }
        ]
    )
    accounts = pd.DataFrame(
        [
            {"account_id": "A", "segment": "retail", "typology": "structuring", "is_suspicious": True},
            {"account_id": "B", "segment": "retail", "typology": "none", "is_suspicious": False},
            {"account_id": "C", "segment": "sme", "typology": "sanctions", "is_suspicious": True},
        ]
    )
    alerts = pd.DataFrame(
        [
            {"account_id": "A", "rule": "structuring", "first_date": pd.Timestamp("2026-01-05"), "detail": "", "severity": "high"},
            {"account_id": "A", "rule": "large_cash", "first_date": pd.Timestamp("2026-01-06"), "detail": "", "severity": "medium"},
            {"account_id": "B", "rule": "large_cash", "first_date": pd.Timestamp("2026-01-07"), "detail": "", "severity": "medium"},
        ]
    )
    path = write_analytics_database(transactions, accounts, alerts, tmp_path / "monitoring.sqlite")
    connection = sqlite3.connect(path)
    yield connection
    connection.close()


def test_sql_views_deduplicate_queue_and_preserve_severity(analytics_database):
    queue = analytics_database.execute(
        "SELECT account_id, alert_count, highest_severity FROM alert_queue ORDER BY account_id"
    ).fetchall()

    assert queue == [("A", 2, "high"), ("B", 1, "medium")]


def test_rule_performance_metrics_match_account_level_labels(analytics_database):
    result = analytics_database.execute(
        "SELECT alerted_accounts, true_positives, false_positives, precision, recall "
        "FROM rule_performance WHERE rule = 'large_cash'"
    ).fetchone()

    assert result[:3] == (2, 1, 1)
    assert result[3:] == pytest.approx((0.5, 0.5))


def test_sql_views_include_unalerted_accounts_and_group_segments(analytics_database):
    outcomes = analytics_database.execute(
        "SELECT account_id, was_alerted FROM account_alert_outcomes ORDER BY account_id"
    ).fetchall()
    segments = analytics_database.execute(
        "SELECT segment, account_count, alerted_accounts FROM segment_performance ORDER BY segment"
    ).fetchall()

    assert outcomes == [("A", 1), ("B", 1), ("C", 0)]
    assert segments == [("retail", 2, 2), ("sme", 1, 0)]


def test_monthly_alert_volume_uses_alert_date(analytics_database):
    months = analytics_database.execute(
        "SELECT alert_month, rule, alerted_accounts FROM monthly_alert_volume "
        "ORDER BY rule"
    ).fetchall()

    assert months == [
        ("2026-01", "large_cash", 2),
        ("2026-01", "structuring", 1),
    ]
