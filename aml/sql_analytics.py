"""Export pipeline data to a SQLite database with reusable analytics views."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def write_analytics_database(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    alerts: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write the pipeline inputs and SQL analytics views to ``path``."""
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        for name, frame in (
            ("transactions", transactions),
            ("accounts", accounts),
            ("alerts", alerts),
        ):
            export = frame.copy()
            for column in export.select_dtypes(include=["datetime", "datetimetz"]).columns:
                export[column] = export[column].astype(str)
            export.to_sql(name, connection, if_exists="replace", index=False)

        sql_path = Path(__file__).with_name("analytics.sql")
        connection.executescript(sql_path.read_text(encoding="utf-8"))

    return database_path
