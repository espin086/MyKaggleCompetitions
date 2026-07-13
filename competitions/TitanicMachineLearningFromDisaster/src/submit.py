"""Submit the current submission to Kaggle and log the public score.

Process (the tracking mechanism for this competition):
  1. Submit submissions/submission.csv via the kaggle CLI.
  2. Poll the submissions endpoint until the score is scored.
  3. Append a row (date, message, public score, status) to SUBMISSIONS.md.

Run from the competition folder:  python src/submit.py "message describing the approach"

Requires Kaggle auth (~/.kaggle/kaggle.json) and the competition rules accepted
on the website, or the CLI returns 401/403.
"""

import csv
import io
import subprocess
import sys
import time
from datetime import datetime, timezone

import config


def _run(args):
    return subprocess.run(args, capture_output=True, text=True)


def submit(message):
    print(f"→ submitting {config.SUBMISSION_PATH} to '{config.SLUG}'")
    r = _run([
        "kaggle", "competitions", "submit", "-c", config.SLUG,
        "-f", config.SUBMISSION_PATH, "-m", message,
    ])
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        sys.exit(f"submit failed: {r.stderr.strip() or r.stdout.strip()}")


def latest_score():
    """Return (status, publicScore) for the most recent submission, or (None, None)."""
    r = _run(["kaggle", "competitions", "submissions", "-c", config.SLUG, "--csv"])
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    if not rows:
        return None, None
    top = rows[0]  # newest first
    return top.get("status"), top.get("publicScore")


def log_result(message, score, status):
    line = (
        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} "
        f"| {message} | {score or 'n/a'} | {status or 'n/a'} |\n"
    )
    header = (
        "# Submissions log\n\n"
        "Public leaderboard scores for each submission, newest at the bottom.\n\n"
        "| Date (UTC) | Message | Public score | Status |\n"
        "|---|---|---|---|\n"
    )
    try:
        with open(config.SUBMISSIONS_LOG) as f:
            content = f.read()
    except FileNotFoundError:
        content = header
    if not content.strip():
        content = header
    with open(config.SUBMISSIONS_LOG, "w") as f:
        f.write(content.rstrip("\n") + "\n" + line)
    print(f"logged to {config.SUBMISSIONS_LOG}: score={score} status={status}")


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python src/submit.py "message describing the approach"')
    message = sys.argv[1]
    submit(message)

    status, score = None, None
    for _ in range(20):  # ~2 min of polling
        time.sleep(6)
        status, score = latest_score()
        if status and status.lower() == "complete" and score:
            break
    log_result(message, score, status)


if __name__ == "__main__":
    main()
