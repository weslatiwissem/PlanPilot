"""Does a snapshot-0 sprint in the training data look like a sprint PlanPilot has just planned?

    python -m agent3_risk.check_serving

The server feeds the model "nothing done, nothing in progress, no scope change, everything still to do". The
model was trained on real snapshot-0 rows. If most of them are like that, the mapping is safe; if not, the
model sees something at serving time that it rarely saw in training.
"""
from typing import Dict

from agent3_risk import sprint_data as sd
from agent3_risk.train_risk import DATA_DIR


def snapshot0_report(s0) -> Dict[str, float]:
    c = s0["commit"]
    return {
        "nothing done yet (vel_done == 0)": float((s0["vel_done"] == 0).mean()),
        "nothing in progress (vel_inprogress == 0)": float((s0["vel_inprogress"] == 0).mean()),
        "no scope added (vel_added == 0)": float((s0["vel_added"] == 0).mean()),
        "no scope removed (vel_removed == 0)": float((s0["vel_removed"] == 0).mean()),
        "all points still to do (vel_todo within 5% of commit)": float(((s0["vel_todo"] - c).abs() <= 0.05 * c).mean()),
        "all issues still to do (no_issuetodo == no_issue_starttime)": float((s0["no_issuetodo"] == s0["no_issue_starttime"]).mean()),
    }


def main() -> None:
    s = sd.samples(sd.load_all(DATA_DIR))
    s0 = s[s["snap"] == 0]
    print(f"{len(s0)} sprints at snapshot 0")
    for name, share in snapshot0_report(s0).items():
        print(f"  {share:>6.1%}  {name}")
    print("Around 90% or more on every line: the planning-time mapping matches the training data.")


if __name__ == "__main__":
    main()
