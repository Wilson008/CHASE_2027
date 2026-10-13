#!/usr/bin/env python3
"""
Eligibility screening for "Who Guards the Gatekeepers?" using CodAGE as the sampling frame.

Produces an eligibility FUNNEL (how many repositories survive each filter) plus a
validation sample for manual precision checking. The funnel numbers are what go
into the Research Design section.

Two stages
----------
Stage 1 (CodAGE only, local):  high-confidence agent-authored PRs -> per-repo
        agent exposure (first agent PR = t0, number of agent PRs, months active).
Stage 2 (GitHub API):          repository metadata (fork / archived / stars) and
        human merge activity in a pre-window and post-window around t0.

Usage
-----
  pip install duckdb requests
  # Step 0: inspect the real schema first and adjust COLS below if needed
  python codage_eligibility.py schema
  # Stage 1 (use a local copy of the parquet; remote reads hit HF rate limits)
  python codage_eligibility.py stage1 --pr-parquet codage/Data/PullRequestEvent.parquet --out out/
  # Precision check of the authorship rules on the stratified sample
  export GITHUB_TOKEN=ghp_...
  python codage_eligibility.py validate --out out/
  # Stage 2 (needs a GitHub token; uses the Search API, ~30 requests/min)
  export GITHUB_TOKEN=ghp_...
  python codage_eligibility.py stage2 --out out/ --metadata-only   # ~1h, no Search API
  python codage_eligibility.py stage2 --out out/                   # later: adds activity counts

By default the PullRequestEvent parquet is read remotely via DuckDB's hf:// support
(only the projected columns are fetched). If that is slow, download the file once
and pass --pr-parquet /path/to/PullRequestEvent.parquet.

All thresholds are command-line parameters; defaults are a starting point, not a
justified choice. Report the values you settle on in the paper.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import duckdb

HF_PR_PARQUET = "hf://datasets/taher-ghaleb/CodAGE/Data/PullRequestEvent.parquet"
CODAGE_END = "2026-04-15"

# ---------------------------------------------------------------------------
# Column names. The five shared columns are documented in the CodAGE card.
# The payload columns below follow the card's naming ("pull_request.title" etc.)
# but are NOT verified -- run `schema` first and fix this mapping if needed.
# Set a value to None if the column does not exist.
# ---------------------------------------------------------------------------
COLS = {
    "repo": "repo_name",
    "ts": "created_at",
    "actor": "actor_login",
    "agent": "agent_type",
    "action": "action",
    "number": "pull_request.number",
    # From Oct 2025 on, GH Archive PullRequestEvent payloads have empty
    # pull_request.user.login and pull_request.body (verified: 100% empty from
    # 2025-11). For an "opened" event the actor IS the PR author, so we use
    # actor_login (a top-level field) instead. Setting author to None makes
    # rule_sql fall back to the actor column.
    "author": None,
    "head_ref": "pull_request.head.ref",
    "title": "pull_request.title",
    "body": "pull_request.body",   # kept for inspection only; NOT used by any rule
}

# ---------------------------------------------------------------------------
# Authorship rules. Constraint: every rule must use ONLY fields that are populated
# across the whole window (actor_login, head.ref), so the detector is
# time-invariant. A body-based rule would silently stop firing after Oct 2025 and
# fabricate a drop in agent activity -- exactly the ingestion artifact we must
# avoid in a longitudinal design. Consequence: Claude Code (body-only signature)
# cannot be detected and is excluded. Rules marked "validate" are branch-only and
# must be precision-checked with the `validate` command before use.
# Review-only agents (CodeRabbit, Gemini Code Assist, PR-Agent) are excluded.
# ---------------------------------------------------------------------------
RULES = [
    # name,          kind,           login(lower) or (branch_prefix, login)
    ("Devin",        "login",        "devin-ai-integration[bot]"),
    ("Jules",        "login",        "google-labs-jules[bot]"),
    ("Jules",        "login",        "jules-google[bot]"),
    ("AmazonQ",      "login",        "amazon-q-developer[bot]"),
    ("Copilot",      "branch_login", ("copilot/", "copilot")),
    ("Codex",        "branch",       "codex/"),       # validate
    ("Cursor",       "branch",       "cursor/"),      # validate
]

SELF_REFERENTIAL_DENYLIST = {
    "princeton-nlp/swe-agent", "swe-agent/swe-agent", "aws/amazon-q-developer-cli",
    "anthropics/claude-code", "openai/codex", "getcursor/cursor", "cognitionlabs/devin",
}


def q(col):
    """Quote a column name for DuckDB (handles dots)."""
    return '"' + col.replace('"', '""') + '"'


def connect():
    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
    except Exception as e:  # local-file mode still works without httpfs
        print(f"[warn] httpfs not loaded: {e}", file=sys.stderr)
    return con


def cmd_schema(args):
    con = connect()
    src = args.pr_parquet or HF_PR_PARQUET
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()
    present = {r[0] for r in rows}
    print(f"Schema of {src}:")
    for r in rows:
        print(f"  {r[0]:<45} {r[1]}")
    print("\nMapping check:")
    for k, v in COLS.items():
        status = "OK" if v in present else ("unset" if v is None else "MISSING -> edit COLS")
        print(f"  {k:<10} {str(v):<35} {status}")


def rule_sql(available):
    """Build a CASE expression assigning a rule name, using only available columns."""
    actor = f"lower({q(COLS['actor'])})"
    author = f"lower({q(COLS['author'])})" if COLS["author"] in available else actor
    ref = f"lower(coalesce({q(COLS['head_ref'])}, ''))" if COLS["head_ref"] in available else "''"
    body = f"lower(coalesce({q(COLS['body'])}, ''))" if COLS["body"] in available else "''"
    whens = []
    for name, kind, spec in RULES:
        if kind == "login":
            whens.append(f"WHEN {author} = '{spec}' THEN '{name}:login'")
        elif kind == "branch_login":
            pref, login = spec
            whens.append(f"WHEN {ref} LIKE '{pref}%' AND {author} = '{login}' THEN '{name}:branch+login'")
        elif kind == "branch":
            whens.append(f"WHEN {ref} LIKE '{spec}%' THEN '{name}:branch'")
        elif kind == "branch_body":
            pref, sub = spec
            whens.append(f"WHEN {ref} LIKE '{pref}%' AND {body} LIKE '%{sub}%' THEN '{name}:branch+body'")
        elif kind == "body":
            whens.append(f"WHEN {body} LIKE '%{spec}%' THEN '{name}:body'")
    return "CASE " + " ".join(whens) + " ELSE NULL END"


def cmd_stage1(args):
    os.makedirs(args.out, exist_ok=True)
    con = connect()
    src = args.pr_parquet or HF_PR_PARQUET
    available = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}')").fetchall()}
    for k in ("repo", "ts", "actor"):
        if COLS[k] not in available:
            sys.exit(f"Required column {COLS[k]} missing; run `schema` and fix COLS.")

    proj = [c for c in COLS.values() if c and c in available]
    sel = ", ".join(q(c) for c in proj)
    action_filter = ""
    if COLS["action"] in available:
        action_filter = f"WHERE lower({q(COLS['action'])}) = 'opened'"
    else:
        print("[warn] no action column: deduplicating events per PR instead", file=sys.stderr)

    # One row per PR: dedupe on (repo, number) if available, else (repo, head_ref, title)
    if COLS["number"] in available:
        key = f"{q(COLS['repo'])}, {q(COLS['number'])}"
    else:
        key = ", ".join(q(c) for c in (COLS["repo"], COLS["head_ref"], COLS["title"]) if c in available)

    t0 = time.time()
    con.execute(f"""
        CREATE TABLE prs AS
        SELECT * FROM (
            SELECT {sel},
                   {rule_sql(available)} AS rule,
                   row_number() OVER (PARTITION BY {key} ORDER BY {q(COLS['ts'])}) AS rn
            FROM read_parquet('{src}')
            {action_filter}
        ) WHERE rn = 1
    """)
    total = con.execute("SELECT count(*) FROM prs").fetchone()[0]
    con.execute("CREATE TABLE agent_prs AS SELECT * FROM prs WHERE rule IS NOT NULL")
    print(f"[stage1] loaded {total:,} candidate PRs in {time.time()-t0:.0f}s")

    # Rule-level counts and a validation sample (for manual precision checking)
    rule_counts = con.execute("SELECT rule, count(*) n FROM agent_prs GROUP BY rule ORDER BY n DESC").fetchall()
    sample_cols = ", ".join(q(c) for c in (COLS["repo"], COLS["number"], COLS["ts"], COLS["actor"],
                                           COLS["head_ref"], COLS["title"]) if c in available)
    # Sample stratified by rule x period (before / from Oct 2025), because the
    # payload change could alter a rule's precision across the two periods.
    period = f"CASE WHEN {q(COLS['ts'])} < '2025-10-01' THEN 'pre_2025-10' ELSE 'post_2025-10' END"
    con.execute(f"""
        COPY (SELECT rule, period, {sample_cols} FROM (
                SELECT *, {period} AS period,
                       row_number() OVER (PARTITION BY rule, {period}
                                          ORDER BY hash({q(COLS['repo'])} || {q(COLS['ts'])})) r
                FROM agent_prs) WHERE r <= {args.sample_per_rule})
        TO '{args.out}/validation_sample.csv' (HEADER)
    """)

    # Monthly counts per rule: a time-invariant detector should show no
    # discontinuity at 2025-10 that is not also visible in the other rules.
    monthly = con.execute(f"""
        PIVOT (SELECT substr({q(COLS['ts'])}, 1, 7) AS month, split_part(rule, ':', 1) AS agent FROM agent_prs)
        ON agent USING count(*) GROUP BY month ORDER BY month
    """)
    monthly_df = monthly.fetchall(); monthly_cols = [d[0] for d in monthly.description]
    with open(f"{args.out}/monthly_rule_counts.csv", "w") as f:
        f.write(",".join(monthly_cols) + "\n")
        for row in monthly_df:
            f.write(",".join("" if v is None else str(v) for v in row) + "\n")

    # Per-repo exposure
    con.execute(f"""
        CREATE TABLE repo AS
        SELECT {q(COLS['repo'])} AS repo,
               min(CAST({q(COLS['ts'])} AS TIMESTAMP)) AS t0,
               count(*) AS n_agent_prs,
               count(DISTINCT split_part(rule, ':', 1)) AS n_agents,
               count(DISTINCT date_trunc('month', CAST({q(COLS['ts'])} AS TIMESTAMP))) AS active_months
        FROM agent_prs GROUP BY 1
    """)

    end = datetime.fromisoformat(CODAGE_END)
    latest_t0 = (end - timedelta(days=30 * args.post_months)).date().isoformat()
    deny = ", ".join(f"'{r}'" for r in SELF_REFERENTIAL_DENYLIST)
    funnel = []
    def count(where):
        return con.execute(f"SELECT count(*) FROM repo WHERE {where}").fetchone()[0]
    c0 = "TRUE"
    c1 = c0 + f" AND lower(repo) NOT IN ({deny})"
    c2 = c1 + f" AND n_agent_prs >= {args.min_agent_prs}"
    c3 = c2 + f" AND active_months >= {args.min_active_months}"
    c4 = c3 + f" AND t0 >= TIMESTAMP '{args.earliest_t0}' AND t0 <= TIMESTAMP '{latest_t0}'"
    for label, cond in [
        ("repos with >=1 high-confidence agent PR", c0),
        ("excluding agent self-referential repos", c1),
        (f">= {args.min_agent_prs} agent PRs", c2),
        (f"agent activity in >= {args.min_active_months} distinct months", c3),
        (f"t0 in [{args.earliest_t0}, {latest_t0}] (>= {args.post_months}-month post-window in CodAGE)", c4),
    ]:
        funnel.append({"stage": "S1", "filter": label, "repos": count(cond)})
    con.execute(f"COPY (SELECT * FROM repo WHERE {c4} ORDER BY n_agent_prs DESC) TO '{args.out}/stage1_candidates.csv' (HEADER)")

    meta = {"source": src, "total_candidate_prs": total,
            "rule_counts": [{"rule": r, "prs": n} for r, n in rule_counts],
            "params": vars(args), "funnel": funnel}
    with open(f"{args.out}/funnel.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print_funnel(funnel)
    print("\nPRs per rule:")
    for r, n in rule_counts:
        print(f"  {r:<25} {n:>10,}")
    print("\nAgent PRs per month (check for a break at 2025-10):")
    print("  " + "".join(f"{c:>10}" for c in monthly_cols))
    for row in monthly_df:
        print("  " + "".join(f"{('' if v is None else v):>10}" for v in row))
    print(f"\nWrote {args.out}/stage1_candidates.csv, validation_sample.csv, monthly_rule_counts.csv, funnel.json")


# ---------------------------------------------------------------------------
# Validate: fetch each sampled PR from the GitHub API and record the ground-truth
# signals that CodAGE no longer carries (author login/type, body). The auto-label
# is only a first pass; skim the CSV before trusting the precision figures.
# ---------------------------------------------------------------------------
SIGNATURES = {
    "Devin": ["devin-ai-integration"], "Jules": ["jules"], "AmazonQ": ["amazon-q"],
    "Copilot": ["copilot"], "Codex": ["chatgpt.com/codex", "codex"], "Cursor": ["cursor"],
}


def cmd_validate(args):
    import csv
    import requests
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    rows = list(csv.DictReader(open(f"{args.out}/validation_sample.csv", encoding="utf-8")))
    out_rows = []
    for i, r in enumerate(rows):
        agent = r["rule"].split(":")[0]
        resp = gh_get(s, f"https://api.github.com/repos/{r[COLS['repo']]}/pulls/{r[COLS['number']]}")
        rec = {"rule": r["rule"], "period": r["period"], "repo": r[COLS["repo"]], "number": r[COLS["number"]],
               "http": resp.status_code, "author": "", "author_type": "", "body_head": "", "auto_label": "unknown"}
        if resp.status_code == 200:
            pr = resp.json()
            user = pr.get("user") or {}
            body = (pr.get("body") or "")
            rec.update(author=user.get("login", ""), author_type=user.get("type", ""),
                       body_head=body[:300].replace("\n", " "))
            sig_in_body_or_login = any(sg in (user.get("login", "") + " " + body).lower() for sg in SIGNATURES[agent])
            rec["auto_label"] = "agent" if (user.get("type") == "Bot" or sig_in_body_or_login) else "check_manually"
        out_rows.append(rec)
        if i % 20 == 0:
            print(f"  {i+1}/{len(rows)}")
    with open(f"{args.out}/validation_checked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print("\nAuto-label summary (agent / check_manually / unknown) per rule x period:")
    from collections import Counter
    c = Counter((r["rule"], r["period"], r["auto_label"]) for r in out_rows)
    keys = sorted({(k[0], k[1]) for k in c})
    for k in keys:
        a, m, u = c[(*k, "agent")], c[(*k, "check_manually")], c[(*k, "unknown")]
        n = a + m
        prec = f"{a/n:.0%}" if n else "n/a"
        print(f"  {k[0]:<22}{k[1]:<14} agent={a:<4} check={m:<4} unreachable={u:<4} auto-precision={prec}")
    print(f"\nWrote {args.out}/validation_checked.csv")


# ---------------------------------------------------------------------------
# Stage 2: GitHub API
# ---------------------------------------------------------------------------
def gh_get(session, url, params=None):
    while True:
        r = session.get(url, params=params, timeout=30)
        if r.status_code in (403, 429) and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 5) + 1
            print(f"  [rate limit] sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        return r


def merged_pr_count(session, repo, start, end):
    """Merged PRs in [start, end) whose author is not a bot (Search API total_count)."""
    query = f"repo:{repo} is:pr is:merged merged:{start}..{end} -author:app/copilot-swe-agent -author:app/devin-ai-integration"
    r = gh_get(session, "https://api.github.com/search/issues", {"q": query, "per_page": 1})
    if r.status_code != 200:
        return None
    time.sleep(2.1)  # Search API: 30 requests/minute
    return r.json().get("total_count")


def cmd_stage2(args):
    import csv
    import requests
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})

    cand_path = f"{args.out}/stage1_candidates.csv"
    cache_path = f"{args.out}/stage2_cache.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            d = json.loads(line); cache[d["repo"]] = d
    rows = list(csv.DictReader(open(cand_path)))
    print(f"[stage2] {len(rows)} candidates, {len(cache)} cached")

    with open(cache_path, "a") as out:
        for i, row in enumerate(rows):
            repo = row["repo"]
            t0 = datetime.fromisoformat(row["t0"].replace(" ", "T")[:19])
            pre_start = (t0 - timedelta(days=30 * args.pre_months)).date().isoformat()
            post_end = (t0 + timedelta(days=30 * args.post_months)).date().isoformat()
            def passes_metadata(d):
                return d.get("status") == 200 and not d.get("fork") and not d.get("archived") \
                       and (d.get("stars") or 0) >= args.min_stars
            if repo in cache:
                d = cache[repo]
                # Re-use cached metadata; only fill in activity counts if they are
                # now needed (e.g. a previous --metadata-only run).
                if args.metadata_only or "pre_merged" in d or not passes_metadata(d):
                    continue
                name = d.get("renamed_to") or repo
            else:
                r = gh_get(s, f"https://api.github.com/repos/{repo}")
                if r.status_code != 200:
                    d = {"repo": repo, "status": r.status_code}
                else:
                    m = r.json()
                    d = {"repo": repo, "status": 200, "fork": m.get("fork"), "archived": m.get("archived"),
                         "stars": m.get("stargazers_count"), "created_at": m.get("created_at"),
                         "renamed_to": m.get("full_name") if m.get("full_name", "").lower() != repo.lower() else None}
                name = d.get("renamed_to") or repo
            # Search API is the slow part (30/min): only query repos that pass
            # the cheap metadata filters, and only when activity counts are wanted.
            if not args.metadata_only and passes_metadata(d):
                d["pre_merged"] = merged_pr_count(s, name, pre_start, t0.date().isoformat())
                d["post_merged"] = merged_pr_count(s, name, t0.date().isoformat(), post_end)
            cache[repo] = d
            out.write(json.dumps(d) + "\n"); out.flush()
            if i % 25 == 0:
                print(f"  {i+1}/{len(rows)} {repo}")

    # Funnel
    funnel = json.load(open(f"{args.out}/funnel.json"))["funnel"]
    def keep(d, step):
        if d.get("status") != 200: return False
        if step >= 1 and (d.get("fork") or d.get("archived")): return False
        if step >= 2 and (d.get("stars") or 0) < args.min_stars: return False
        if step >= 3 and (d.get("pre_merged") or 0) < args.min_pre_merged: return False
        if step >= 4 and (d.get("post_merged") or 0) < args.min_post_merged: return False
        return True
    labels = ["still reachable via API", "not a fork, not archived", f">= {args.min_stars} stars",
              f">= {args.min_pre_merged} merged PRs in {args.pre_months}-month pre-window",
              f">= {args.min_post_merged} merged PRs in {args.post_months}-month post-window"]
    last_step = 2 if args.metadata_only else 4
    labels = labels[: last_step + 1]
    funnel = [f for f in funnel if f["stage"] == "S1"]
    for step, label in enumerate(labels):
        funnel.append({"stage": "S2", "filter": label,
                       "repos": sum(keep(cache[r["repo"]], step) for r in rows if r["repo"] in cache)})
    eligible = [cache[r["repo"]] | r for r in rows if r["repo"] in cache and keep(cache[r["repo"]], last_step)]
    with open(f"{args.out}/eligible_repos.csv", "w", newline="") as f:
        if eligible:
            w = csv.DictWriter(f, fieldnames=list(eligible[0].keys())); w.writeheader(); w.writerows(eligible)
    meta = json.load(open(f"{args.out}/funnel.json")); meta["funnel"] = funnel; meta["stage2_params"] = vars(args)
    json.dump(meta, open(f"{args.out}/funnel.json", "w"), indent=2, default=str)
    print_funnel(funnel)
    print(f"\nWrote {args.out}/eligible_repos.csv ({len(eligible)} repos)")


def print_funnel(funnel):
    print("\nEligibility funnel")
    print(f"  {'stage':<6}{'filter':<75}{'repos':>8}")
    for f in funnel:
        print(f"  {f['stage']:<6}{f['filter']:<75}{f['repos']:>8,}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("schema", "stage1", "validate", "stage2"):
        sp = sub.add_parser(name)
        sp.add_argument("--pr-parquet", help="local PullRequestEvent.parquet (default: read from Hugging Face)")
        sp.add_argument("--out", default="out")
        sp.add_argument("--post-months", type=int, default=6)
        sp.add_argument("--pre-months", type=int, default=6)
        if name == "stage1":
            sp.add_argument("--min-agent-prs", type=int, default=20)
            sp.add_argument("--min-active-months", type=int, default=3)
            sp.add_argument("--earliest-t0", default="2025-01-01")
            sp.add_argument("--sample-per-rule", type=int, default=30)
        if name == "stage2":
            sp.add_argument("--min-stars", type=int, default=100)
            sp.add_argument("--min-pre-merged", type=int, default=30)
            sp.add_argument("--min-post-merged", type=int, default=30)
            sp.add_argument("--metadata-only", action="store_true",
                            help="skip the slow Search API activity counts; funnel stops at the stars filter")
    args = p.parse_args()
    {"schema": cmd_schema, "stage1": cmd_stage1, "validate": cmd_validate, "stage2": cmd_stage2}[args.cmd](args)


if __name__ == "__main__":
    main()
