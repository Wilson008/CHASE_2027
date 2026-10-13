import duckdb
con = duckdb.connect()
src = "codage/Data/PullRequestEvent.parquet"
con.sql(f"""
SELECT substr(created_at,1,7) AS month, count(*) AS n,
  round(avg((coalesce("pull_request.body",'')='')::INT),3)       AS body_empty,
  round(avg((coalesce("pull_request.user.login",'')='')::INT),3) AS author_empty,
  round(avg((coalesce("pull_request.head.ref",'')='')::INT),3)   AS ref_empty
FROM read_parquet('{src}')
WHERE action = 'opened'
GROUP BY 1 ORDER BY 1
""").show(max_rows=40)