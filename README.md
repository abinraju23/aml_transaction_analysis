# Transaction monitoring: what the rules cost, and what they miss

A rule-based anti-money-laundering monitoring system over a synthetic transaction book, and —
the actual point of the project — an honest measurement of the trade-off every one of its thresholds
makes between **analyst hours spent** and **reportable cases missed**.

This is not a classifier. There is no accuracy score anywhere in the output, because accuracy is the
wrong unit for a problem where one error type costs twenty minutes and the other is a regulatory
breach.

```bash
pip install -r requirements.txt
pytest                                      # 30+ tests, no data download needed
python -m aml.run_pipeline                  # writes aml/results/monitoring_report.md
python -m aml.run_pipeline --sweep pass_through_ratio
```

---

## Why this problem, in this form

Transaction monitoring is an asymmetric decision problem dressed up as a classification problem.

- A **false positive** costs real money every single time: an analyst opens the alert, pulls the
  customer history, and closes it. At twenty minutes an alert that is about EUR 11 of salary, and a
  mid-size Irish bank runs tens of thousands of alerts a year.
- A **false negative** is rare and potentially catastrophic: an unreported case, a lookback exercise,
  and in the sanctions case strict liability — the breach is the payment itself, not a judgement
  about whether it looked suspicious.

Those two are not in the same units, so there is no single threshold that is "correct". There is only
a trade-off, and the professional skill is being able to state the exchange rate you assumed and
defend the setting you chose. That is what this repository produces: a table where every row is a
full re-run of the rule set at a different threshold, priced in euro, with the assumptions written
above it.

## The rules

Five rules, each with explicit thresholds, each returning account-level alerts with the reason
attached (`aml/rules.py`):

| Rule | What it looks for | Why it is here |
|---|---|---|
| `large_cash` | A single cash credit at or above the reporting threshold | The crudest rule in any system, and the one that generates the most false positives, because plenty of legitimate businesses bank large cash. Kept precisely so the report can show that. |
| `structuring` | Several cash credits *just under* the threshold within a short window | Catches what `large_cash` is designed to miss. The interesting parameter is the band: too narrow and a launderer one euro lower walks through; too wide and every ordinary four-figure deposit alerts. |
| `pass_through` | Money in, 85%+ of it back out within days | A funnel account seen from the bank's side: high turnover, negligible balance, no economic purpose on its face. |
| `dormant_reactivation` | Months of silence, then real volume | A classic mule indicator with an obvious false positive — an account used once a year for a tax bill looks identical. |
| `sanctions_screening` | A counterparty on the screening list, after name normalisation | The one rule where a miss is not a judgement call, which is why the cost model prices it at ten times the others. |

Thresholds live in a single frozen `Thresholds` dataclass, so a sweep is a loop over
`dataclasses.replace` and no run can quietly inherit the settings of the run before it.

## The vocabulary, used the way the industry uses it

A rule fires an **alert**; an analyst triages alerts into **cases**; a case that survives review
becomes a report to the FIU and Revenue (the Irish equivalent of a SAR). This repository models the
first step only. The reading behind it is public and free: the Central Bank of Ireland's AML/CFT
guidelines for the financial sector, and the FATF 40 Recommendations — customer due diligence,
transaction monitoring, the alert-to-case-to-report funnel, PEP and sanctions screening, threshold
tuning, SLA adherence.

## The data

`aml/data.py` generates a deterministic synthetic book — accounts, business-day transactions, and
four injected typologies (structuring, pass-through, dormant reactivation, sanctions exposure) at a
realistic base rate of **under 3% of accounts**. A rule that looks good on a balanced sample is
worthless, so the generator does not produce one.

Ordinary customers are deliberately given a small share of genuinely large cash deposits, which is
what makes the naive threshold rule produce false positives rather than a clean separation. Without
that, the whole exercise would flatter itself.

`load_public_dataset()` normalises column names for the public sets worth moving to next, in order:

1. **Elliptic Data Set** — bitcoin transaction graph, labelled licit/illicit. A graph problem, which
   reuses the Node2Vec graph-embedding work from my dissertation rather than teaching a new skill.
2. **IBM Transactions for Anti Money Laundering** (Kaggle) — synthetic but purpose-built, and the
   closest public thing to what a monitoring system actually sees.
3. **PaySim** (Kaggle) — synthetic mobile money, the standard fraud teaching set.

The rules and the evaluation do not care where the rows came from, so that move is a loader change
and nothing else.

## What the output contains

`aml/results/monitoring_report.md`, generated:

1. **Per-rule performance** — alert volume, precision, recall against the whole suspicious
   population *and* recall against the typology the rule was actually written to catch. Judging a
   sanctions rule on its structuring recall is meaningless, so both are reported.
2. **The queue an analyst actually faces** — de-duplicated to accounts, because the per-rule rows
   double-count any customer that three rules alert on, and quoting only per-rule precision hides
   that.
3. **What the rule set missed**, grouped by typology.
4. **A threshold sweep** — every row a full re-run, with alert volume, analyst hours, missed cases,
   missed sanctions cases, and total expected cost in euro, plus the chart.
5. **The recommendation**, stated as a conditional: on *these* cost assumptions the cheapest setting
   is X, and if you price a missed case differently the answer moves.
6. **A queryable SQLite database** with the source transactions, account labels, and generated alerts,
   plus views for the de-duplicated queue, rule performance, monthly alert volume, and segment
   performance. This provides a SQL-first way to inspect the same monitoring run.

### Explore the SQL output

Each pipeline run writes `aml/results/monitoring.sqlite`. Open it with SQLite, DBeaver, or another
SQL client and query the views directly:

```sql
SELECT rule, alerted_accounts, true_positives, false_positives, precision, recall
FROM rule_performance
ORDER BY precision DESC;

SELECT segment, account_count, alerted_accounts, alert_rate, precision
FROM segment_performance
ORDER BY alert_rate DESC;

SELECT alert_month, rule, alerted_accounts
FROM monthly_alert_volume
ORDER BY alert_month, rule;
```

The account-level `account_alert_outcomes` view de-duplicates customers alerted by multiple rules.
Labels are included for this synthetic evaluation only; they would not be available for live customers.
The SQL layer uses Python's standard-library SQLite support, so it adds no package dependency.

## Cost assumptions

Every one of these is arguable, which is why they are command-line arguments rather than constants:

| Assumption | Default | Basis |
|---|---|---|
| Triage time per alert | 20 minutes | Typical first-line review of a single-rule alert |
| Analyst cost | EUR 32/hour | Roughly EUR 45k fully loaded |
| Cost of a missed reportable case | EUR 25,000 | Expected remediation and lookback exposure, not a prediction of a fine |
| Cost of a missed sanctions hit | EUR 250,000 | Strict liability, so categorically worse |

## The limitation that matters most

**Recall is measurable here only because the data is synthetic and the typologies were injected
deliberately.** On a real book nobody knows the denominator: you can measure the cases you found and
never the ones you did not. Every recall figure in the output should be read as *recall against known
injected typologies*, which is an upper bound on what the same rules would achieve against an
adversary who is adapting to them.

Two more, stated rather than buried:

- **Screening here is exact-match after normalisation.** Real sanctions screening is fuzzy-matched,
  transliteration-aware and tuned against its own false-positive problem. This is a demonstration of
  where screening sits in the pipeline, not a screening engine.
- **Rules are univariate and static.** Real systems combine them, score customers, and feed
  behavioural baselines per segment. Adding that would improve the detection rate and would not
  change the shape of the cost argument, which is the thing being demonstrated.

## What this project does and does not establish

It establishes that I can take a monitoring problem, write the rules, measure the alert volume they
generate, price the asymmetry between a false positive and a missed case, and defend a threshold
choice with a table rather than an opinion. It makes an interest in financial crime analytics a true
statement rather than an aspirational one.

It does **not** substitute for having worked in a regulated environment, which is what most of these
employers actually screen on, and it is not a compliance qualification. It moves the domain criterion
from an automatic fail to arguable. Nothing more.

---

## Layout

```
aml/
  data.py          synthetic book with injected typologies, plus public-dataset loader
  rules.py         five monitoring rules, all thresholds in one frozen dataclass
  evaluate.py      confusion counts, per-rule and portfolio metrics, cost model, threshold sweep
  report.py        markdown tables and the sweep chart
  analytics.sql    SQLite views for queue, rule, monthly, and segment analysis
  sql_analytics.py SQLite export used by the pipeline
  run_pipeline.py  CLI
  tests/           test_rules.py (every rule on hand-built cases), test_evaluate.py (the arithmetic)
  results/         generated
```

Educational and research use only. The synthetic data describes nobody, and nothing here is a
compliance control.
