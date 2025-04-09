SELECT 
    c.name AS campaign_name,
    SUM(e.emails_sent) AS total_emails_sent,
    SUM(e.emails_opened) AS total_emails_opened,
    SUM(e.emails_sent - e.emails_opened) AS total_emails_not_opened
FROM campaigns c
JOIN email_stats e ON c.id = e.campaign_id
GROUP BY c.name
ORDER BY c.name ASC;