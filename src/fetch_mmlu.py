"""Download a fixed MMLU subset for the capability sanity check.

Pulls the original Hendrycks tarball directly (~7 MB) and extracts the test
split for a hand-picked set of subjects. Avoids the `datasets` dependency for
the same Windows segfault reason as build_data.py.

Outputs a single JSONL at data/mmlu_test.jsonl with one row per question:
    {"subject": str, "question": str, "choices": [A, B, C, D], "answer": "A"|"B"|"C"|"D"}
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
import urllib.request
from pathlib import Path


MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

# Five subjects: a mix of reasoning, STEM, and humanities. This gives ~500
# questions, plenty for detecting a 2–3 pp capability drop with confidence.
SUBJECTS: tuple[str, ...] = (
    "formal_logic",
    "college_computer_science",
    "high_school_world_history",
    "philosophy",
    "professional_psychology",
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LETTERS = ("A", "B", "C", "D")


def main() -> None:
    print(f"downloading MMLU tarball ({MMLU_URL})...")
    raw = urllib.request.urlopen(MMLU_URL).read()
    print(f"  got {len(raw):,} bytes")

    items: list[dict] = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tar:
        for subject in SUBJECTS:
            member_name = f"data/test/{subject}_test.csv"
            try:
                member = tar.getmember(member_name)
            except KeyError as e:
                raise RuntimeError(f"missing in tarball: {member_name}") from e
            f = tar.extractfile(member)
            assert f is not None
            text = f.read().decode("utf-8")
            rows = list(csv.reader(io.StringIO(text)))
            for row in rows:
                # Format: question, A, B, C, D, answer_letter
                if len(row) != 6:
                    continue
                q, a, b, c, d, ans = row
                ans = ans.strip().upper()
                if ans not in LETTERS:
                    continue
                items.append({
                    "subject": subject,
                    "question": q,
                    "choices": [a, b, c, d],
                    "answer": ans,
                })

    out = DATA_DIR / "mmlu_test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for it in items:
            fp.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} questions across {len(SUBJECTS)} subjects to {out}")


if __name__ == "__main__":
    main()
