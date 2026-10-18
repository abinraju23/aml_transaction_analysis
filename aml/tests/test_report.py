from __future__ import annotations

import pandas as pd

from aml.report import build_dashboard_html


def test_build_dashboard_html_contains_summary_and_sweep_sections():
    summary = {
        "accounts_reviewed": 100,
        "precision": 0.2,
        "recall": 0.9,
        "total_expected_cost_eur": 123456,
    }
    rules_table = pd.DataFrame([
        {
            "rule": "large_cash",
            "alerts": 10,
            "precision": 0.2,
            "recall": 0.5,
            "target_recall": 0.8,
            "target_population": 4,
            "alerts_per_true_case": 2.0,
        }
    ])
    missed = pd.DataFrame([
        {"typology": "pass_through", "population": 4, "detected": 3, "missed": 1, "detection_rate": 0.75}
    ])
    sweep = pd.DataFrame([
        {"large_cash": 10000, "accounts_reviewed": 100, "total_expected_cost_eur": 123456, "recall": 0.9},
        {"large_cash": 15000, "accounts_reviewed": 80, "total_expected_cost_eur": 220000, "recall": 0.7},
    ])

    html = build_dashboard_html(summary, rules_table, missed, sweep, "large_cash", "Z:/tmp/report.html")

    assert "AML Decision Lens" in html
    assert "Summary" in html
    assert "Threshold sweep" in html
    assert "Preferred threshold" in html
    assert "large_cash" in html
    assert "123,456" in html
