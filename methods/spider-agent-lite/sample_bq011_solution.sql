-- This is a sample solution for the bq011 example
-- Task: Find how many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021

WITH active_users_last_7_days AS (
  -- Users who had events in the 7-day period
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201231' AND '20210106'
),
active_users_last_2_days AS (
  -- Users who had events in the last 2 days
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210105' AND '20210106'
)
SELECT 
  COUNT(DISTINCT a.user_pseudo_id) AS n_day_inactive_users_count
FROM active_users_last_7_days a
LEFT JOIN active_users_last_2_days b
  ON a.user_pseudo_id = b.user_pseudo_id
WHERE b.user_pseudo_id IS NULL
