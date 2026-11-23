"""Synthetic sprint snapshots in the TSE2017 file format, so tests and demos run without the real data.

    python -m agent3_risk.synthetic data/synthetic_tse2017

The numbers are random. They exist to exercise the code (loading, features, CV plumbing). Never report a
metric computed on them.
"""
import sys
from pathlib import Path

import numpy as np

HEADER = ["boardid", "sprintid", "name", "planday", "no_issue_starttime", "vel_starttime", "no_issue_added",
          "vel_added", "no_issue_removed", "vel_removed", "no_issuetodo", "vel_todo", "no_issueinprogress",
          "vel_inprogress", "no_issuedone", "vel_done", "no_admin", "no_teammember", "vel_diff"]
FRACTION = {0: 0.0, 30: 0.3, 50: 0.5, 80: 0.8}


def write_synthetic(out_dir, boards: int = 40, sprints_per_board: int = 9, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    rows = {snap: [] for snap in FRACTION}
    sprint_id = 1000
    for board in range(1, boards + 1):
        capacity = rng.lognormal(3.0, 0.4)  # the board's usual velocity
        team, admin = int(rng.integers(2, 8)), int(rng.integers(1, 3))
        planday = int(rng.choice([7, 14, 21]))
        for j in range(sprints_per_board):
            sprint_id += 1
            commit0 = capacity * rng.uniform(0.6, 1.7)
            final_done = capacity * rng.uniform(0.55, 1.25)
            added_end, removed_end = commit0 * rng.uniform(0, 0.25), commit0 * rng.uniform(0, 0.15)
            name = f"sprint {j}" if rng.random() > 0.1 else f"team-x, sprint {j}"  # unquoted comma, as in the real files
            for snap, frac in FRACTION.items():
                added, removed = added_end * min(1.0, frac * 1.3), removed_end * min(1.0, frac * 1.3)
                commit_t = commit0 + added - removed
                done = final_done * frac * rng.uniform(0.8, 1.2)
                inprog = max(0.0, (commit_t - done) * rng.uniform(0.1, 0.4))
                todo = max(0.0, commit_t - done - inprog)

                def n(v):
                    return max(0, int(round(v / 3)))

                values = [planday, n(commit0), round(commit0, 1), n(added), round(added, 1), n(removed),
                          round(removed, 1), n(todo), round(todo, 1), n(inprog), round(inprog, 1), n(done),
                          round(done, 1), admin, team, round(final_done - commit_t, 1)]
                rows[snap].append(",".join([str(board), str(sprint_id), name] + [str(v) for v in values]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for snap, lines in rows.items():
        (out / f"jira_iteration_{snap}.csv").write_text(
            ",".join(HEADER) + "\n" + "\n".join(lines) + "\n", encoding="latin-1")


if __name__ == "__main__":
    write_synthetic(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_tse2017")
    print("written")
