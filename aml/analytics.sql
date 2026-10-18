CREATE VIEW IF NOT EXISTS alert_queue AS
SELECT
    account_id,
    COUNT(*) AS alert_count,
    GROUP_CONCAT(DISTINCT rule) AS rules,
    CASE MAX(CASE severity
        WHEN 'low' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'high' THEN 3
        WHEN 'critical' THEN 4
        ELSE 0
    END)
        WHEN 1 THEN 'low'
        WHEN 2 THEN 'medium'
        WHEN 3 THEN 'high'
        WHEN 4 THEN 'critical'
        ELSE NULL
    END AS highest_severity,
    MIN(first_date) AS first_alert_date
FROM alerts
GROUP BY account_id;

CREATE VIEW IF NOT EXISTS account_alert_outcomes AS
SELECT
    a.account_id,
    a.segment,
    a.typology,
    a.is_suspicious,
    COALESCE(q.alert_count, 0) AS alert_count,
    q.rules,
    q.highest_severity,
    q.first_alert_date,
    CASE WHEN q.account_id IS NULL THEN 0 ELSE 1 END AS was_alerted
FROM accounts AS a
LEFT JOIN alert_queue AS q ON q.account_id = a.account_id;

CREATE VIEW IF NOT EXISTS rule_performance AS
SELECT
    al.rule,
    COUNT(DISTINCT al.account_id) AS alerted_accounts,
    COUNT(DISTINCT CASE WHEN ac.is_suspicious = 1 THEN al.account_id END) AS true_positives,
    COUNT(DISTINCT CASE WHEN ac.is_suspicious = 0 THEN al.account_id END) AS false_positives,
    1.0 * COUNT(DISTINCT CASE WHEN ac.is_suspicious = 1 THEN al.account_id END)
        / NULLIF(COUNT(DISTINCT al.account_id), 0) AS precision,
    1.0 * COUNT(DISTINCT CASE WHEN ac.is_suspicious = 1 THEN al.account_id END)
        / NULLIF((SELECT COUNT(*) FROM accounts WHERE is_suspicious = 1), 0) AS recall
FROM alerts AS al
JOIN accounts AS ac ON ac.account_id = al.account_id
GROUP BY al.rule;

CREATE VIEW IF NOT EXISTS monthly_alert_volume AS
SELECT
    strftime('%Y-%m', first_date) AS alert_month,
    rule,
    COUNT(DISTINCT account_id) AS alerted_accounts
FROM alerts
GROUP BY strftime('%Y-%m', first_date), rule;

CREATE VIEW IF NOT EXISTS segment_performance AS
SELECT
    segment,
    COUNT(*) AS account_count,
    SUM(was_alerted) AS alerted_accounts,
    SUM(is_suspicious) AS suspicious_accounts,
    1.0 * SUM(was_alerted) / NULLIF(COUNT(*), 0) AS alert_rate,
    1.0 * SUM(CASE WHEN was_alerted = 1 AND is_suspicious = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(was_alerted), 0) AS precision
FROM account_alert_outcomes
GROUP BY segment;
