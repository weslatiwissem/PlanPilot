"""Sanity probe of the saved sprint-risk models: does the forecast rise with the load and with a worse history?

    python -m agent3_risk.probe_forecast

For a team of 3, the same sprint is forecast with growing load (points and stories) under different histories.
A model a user can trust must not say that a sprint of 45 points is safer than one of 10 points. The table
marks every place where the forecast FALLS although the load grows.
"""
from typing import Dict, List, Optional, Tuple

from agent3_risk import serve

LOADS: List[Tuple[float, int]] = [(6, 2), (9, 3), (15, 5), (21, 7), (30, 10), (45, 15)]  # (points, stories): 3 points each
HISTORIES: Dict[str, List[dict]] = {
    "no history": [],
    "reliable: committed 10, delivered 10 (x3)": [{"committed": 10, "delivered": 10}] * 3,
    "weak: committed 10, delivered 6 (x3)": [{"committed": 10, "delivered": 6}] * 3,
    "bigger team velocity: committed 25, delivered 24 (x3)": [{"committed": 25, "delivered": 24}] * 3,
}


def probe(team: int = 3) -> Dict[str, Optional[List[float]]]:
    cfg = serve.load_models()
    if cfg is None:
        raise SystemExit(f"No model in {serve.model_dir()}. Train one: python -m agent3_risk.train_final --save")
    sprints = [{"number": i + 1, "n": n, "points": float(p)} for i, (p, n) in enumerate(LOADS)]
    out = {}
    for label, past in HISTORIES.items():
        X1, _, X2, _ = serve.build_features(sprints, past, team, cfg["impute"]["planday"], cfg["impute"]["no_admin"],
                                            cfg["history"])
        name = serve.served_name(cfg, bool(past))
        out[label] = None if name is None else [float(p) for p in cfg["boosters"][name].predict(X2 if past else X1)]
    return out


def falls(values: List[float], tolerance: float = 0.005) -> int:
    """How many times the forecast drops although the load grew."""
    return sum(1 for a, b in zip(values, values[1:]) if b < a - tolerance)


def main() -> None:
    table = probe()
    print("forecast = chance of delivering less than committed, team of 3, sprint length from the training median")
    print(f"{'history':<56}" + "".join(f"{p:>6g}p" for p, _ in LOADS) + "   drops")
    for label, values in table.items():
        if values is None:
            print(f"{label:<56}  not served (no forecast in this situation)")
            continue
        print(f"{label:<56}" + "".join(f"{v:>7.0%}" for v in values) + f"   {falls(values)}")
    print("\nIdeally every row rises from left to right (0 drops). Each drop is a place where the model says a "
          "heavier sprint is safer.")


if __name__ == "__main__":
    main()
