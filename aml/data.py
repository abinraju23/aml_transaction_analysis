"""Transaction data, synthetic by default.

The project needs labelled transactions: a ground truth of which accounts are
actually laundering, so that a rule can be scored rather than admired. Real AML
data is either confidential or has labels nobody can check, so this module
generates a synthetic book with known typologies injected, and keeps a loader
for the public datasets worth moving to afterwards.

The synthetic generator is deterministic given a seed. That matters because a
threshold-tuning table computed on data that changes between runs is not a
result.

Typologies injected, each a pattern a real transaction-monitoring system is
written to catch:

* **structuring** - a deposit broken into several amounts just under a reporting
  threshold, over a short window.
* **pass-through / funnel** - money in and almost all of it straight out again
  within days, leaving a near-zero balance.
* **dormant reactivation** - an account quiet for months that suddenly moves
  large volume.
* **sanctions exposure** - a counterparty on the screening list.

Public datasets to move to once the rules work, in the order they are worth it:

* **Elliptic Data Set** - bitcoin transaction graph, labelled licit/illicit. A
  graph problem, which reuses the Node2Vec embedding work from the dissertation
  rather than teaching a new skill from scratch.
* **IBM Transactions for Anti Money Laundering** (Kaggle) - synthetic but
  purpose-built, and the closest public thing to what a monitoring system sees.
* **PaySim** (Kaggle) - synthetic mobile money, the standard fraud teaching set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CASH_REPORTING_THRESHOLD = 10_000.0  # the figure the structuring typology hides under
SANCTIONED_COUNTERPARTIES = (
    "ACME TRADING FZE",
    "NORTHSTAR SHIPPING OOO",
    "BLUEWATER HOLDINGS SA",
)


@dataclass(frozen=True)
class BookConfig:
    """Size and composition of the generated book."""

    n_accounts: int = 2_000
    n_days: int = 180
    start: str = "2025-01-06"
    # Share of accounts given each typology. Deliberately small: real base rates
    # are well under 1%, and a rule that looks good on a balanced sample is
    # worthless.
    p_structuring: float = 0.010
    p_pass_through: float = 0.008
    p_dormant: float = 0.006
    p_sanctions: float = 0.004
    seed: int = 20_260_101
    sanctioned: tuple[str, ...] = field(default=SANCTIONED_COUNTERPARTIES)


def generate_book(config: BookConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(transactions, accounts)``.

    ``transactions`` columns: ``txn_id``, ``date``, ``account_id``, ``amount``
    (positive = credit in, negative = debit out), ``channel``, ``counterparty``.

    ``accounts`` columns: ``account_id``, ``opened``, ``segment``, ``is_pep``,
    ``typology`` (``none`` for the honest majority) and ``is_suspicious``, which
    is the label. Nothing downstream may read ``typology`` or ``is_suspicious``
    except the evaluation step - that is the whole discipline of the exercise.
    """
    cfg = config or BookConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=cfg.n_days)

    account_ids = [f"ACC{i:06d}" for i in range(cfg.n_accounts)]
    segments = rng.choice(
        ["retail", "sme", "corporate"], size=cfg.n_accounts, p=[0.75, 0.20, 0.05]
    )
    typology = np.full(cfg.n_accounts, "none", dtype=object)

    # Assign typologies without overlap, so each label means one thing.
    pool = rng.permutation(cfg.n_accounts)
    cursor = 0
    for name, share in (
        ("structuring", cfg.p_structuring),
        ("pass_through", cfg.p_pass_through),
        ("dormant", cfg.p_dormant),
        ("sanctions", cfg.p_sanctions),
    ):
        take = int(round(share * cfg.n_accounts))
        typology[pool[cursor : cursor + take]] = name
        cursor += take

    accounts = pd.DataFrame(
        {
            "account_id": account_ids,
            "opened": [
                dates[0] - pd.Timedelta(days=int(d))
                for d in rng.integers(30, 3_000, cfg.n_accounts)
            ],
            "segment": segments,
            "is_pep": rng.random(cfg.n_accounts) < 0.02,
            "typology": typology,
            "is_suspicious": typology != "none",
        }
    )

    rows: list[dict] = []
    for idx, account_id in enumerate(account_ids):
        segment = segments[idx]
        rows.extend(_baseline_activity(rng, account_id, segment, dates))
        kind = typology[idx]
        if kind == "structuring":
            rows.extend(_structuring(rng, account_id, dates))
        elif kind == "pass_through":
            rows.extend(_pass_through(rng, account_id, dates))
        elif kind == "dormant":
            rows.extend(_dormant_reactivation(rng, account_id, dates))
        elif kind == "sanctions":
            rows.extend(_sanctions_exposure(rng, account_id, dates, cfg.sanctioned))

    txns = pd.DataFrame(rows).sort_values(["date", "account_id"]).reset_index(drop=True)
    txns.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(txns))])
    return txns, accounts


# --------------------------------------------------------------------------- #
# Behaviour generators. Each returns a list of transaction dicts.
# --------------------------------------------------------------------------- #

_SCALE = {"retail": 350.0, "sme": 2_400.0, "corporate": 18_000.0}
_RATE = {"retail": 0.9, "sme": 2.2, "corporate": 5.0}


def _baseline_activity(rng, account_id: str, segment: str, dates) -> list[dict]:
    """Ordinary customers. Lognormal amounts, Poisson daily counts, and a
    minority of legitimately large cash deposits - which is what makes a naive
    threshold rule produce false positives rather than a clean separation."""
    out = []
    n_per_day = rng.poisson(_RATE[segment], size=len(dates))
    for day, count in zip(dates, n_per_day):
        for _ in range(int(count)):
            amount = float(rng.lognormal(np.log(_SCALE[segment]), 1.1))
            is_credit = rng.random() < 0.48
            # ~0.4% of ordinary activity is a genuinely large cash deposit.
            if rng.random() < 0.004:
                amount = float(rng.uniform(9_000, 45_000))
            out.append(
                {
                    "date": day,
                    "account_id": account_id,
                    "amount": round(amount if is_credit else -amount, 2),
                    "channel": str(rng.choice(["card", "transfer", "cash", "direct_debit"],
                                              p=[0.42, 0.38, 0.12, 0.08])),
                    "counterparty": f"CP{int(rng.integers(0, 4000)):05d}",
                }
            )
    return out


def _structuring(rng, account_id: str, dates) -> list[dict]:
    """Several cash credits just under the reporting threshold, inside a week."""
    out = []
    n_episodes = int(rng.integers(1, 3))
    for _ in range(n_episodes):
        start = int(rng.integers(0, max(len(dates) - 8, 1)))
        for offset in range(int(rng.integers(3, 6))):
            out.append(
                {
                    "date": dates[min(start + offset, len(dates) - 1)],
                    "account_id": account_id,
                    "amount": round(float(rng.uniform(0.82, 0.985)) * CASH_REPORTING_THRESHOLD, 2),
                    "channel": "cash",
                    "counterparty": f"CP{int(rng.integers(0, 4000)):05d}",
                }
            )
    return out


def _pass_through(rng, account_id: str, dates) -> list[dict]:
    """A large credit in, then 90-99% of it out within a few days."""
    out = []
    for _ in range(int(rng.integers(2, 5))):
        start = int(rng.integers(0, max(len(dates) - 6, 1)))
        amount = float(rng.uniform(25_000, 180_000))
        out.append(
            {
                "date": dates[start],
                "account_id": account_id,
                "amount": round(amount, 2),
                "channel": "transfer",
                "counterparty": f"CP{int(rng.integers(0, 4000)):05d}",
            }
        )
        remaining = amount * float(rng.uniform(0.90, 0.99))
        for offset in (1, 2, 3):
            slice_ = remaining / 3.0
            out.append(
                {
                    "date": dates[min(start + offset, len(dates) - 1)],
                    "account_id": account_id,
                    "amount": -round(slice_, 2),
                    "channel": "transfer",
                    "counterparty": f"CP{int(rng.integers(0, 4000)):05d}",
                }
            )
    return out


def _dormant_reactivation(rng, account_id: str, dates) -> list[dict]:
    """Nothing for most of the window, then a burst of volume at the end."""
    out = []
    start = int(len(dates) * 0.8)
    for offset in range(int(rng.integers(4, 10))):
        out.append(
            {
                "date": dates[min(start + offset, len(dates) - 1)],
                "account_id": account_id,
                "amount": round(float(rng.uniform(8_000, 60_000)), 2),
                "channel": str(rng.choice(["transfer", "cash"], p=[0.7, 0.3])),
                "counterparty": f"CP{int(rng.integers(0, 4000)):05d}",
            }
        )
    return out


def _sanctions_exposure(rng, account_id: str, dates, sanctioned: tuple[str, ...]) -> list[dict]:
    """Ordinary-looking payments, one or two of them to a listed counterparty."""
    out = []
    for _ in range(int(rng.integers(1, 4))):
        out.append(
            {
                "date": dates[int(rng.integers(0, len(dates)))],
                "account_id": account_id,
                "amount": -round(float(rng.uniform(3_000, 40_000)), 2),
                "channel": "transfer",
                "counterparty": str(rng.choice(sanctioned)),
            }
        )
    return out


def load_public_dataset(path: str) -> pd.DataFrame:
    """Read a public AML dataset from CSV and normalise its column names.

    Kept deliberately thin: the point is that the rules and the evaluation do
    not care where the rows came from, so moving from synthetic data to the IBM
    or PaySim set is a loader change and nothing else.
    """
    df = pd.read_csv(path)
    renames = {
        "Timestamp": "date",
        "step": "date",
        "Account": "account_id",
        "nameOrig": "account_id",
        "Amount Received": "amount",
        "amount": "amount",
        "nameDest": "counterparty",
        "Receiving Currency": "currency",
        "Payment Format": "channel",
        "type": "channel",
        "isFraud": "is_suspicious",
        "Is Laundering": "is_suspicious",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing = {"date", "account_id", "amount"} - set(df.columns)
    if missing:
        raise ValueError(f"dataset at {path} is missing required columns: {sorted(missing)}")
    return df
