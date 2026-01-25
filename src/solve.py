# src/solve.py
# src/solve.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pyomo.environ as pyo
from pyomo.opt.base.solvers import OptSolver  # ✅ proper type

from .data_loader import load_instance
from .build_model import build_model


def pick_solver() -> OptSolver:
    """Pick a MILP solver (prefer HiGHS via appsi)."""
    try:
        s = pyo.SolverFactory("appsi_highs")
        if s.available():
            return s
    except Exception:
        pass

    try:
        s = pyo.SolverFactory("highs")
        if s.available():
            return s
    except Exception:
        pass

    raise RuntimeError(
        "No suitable MILP solver found. Install HiGHS via: pip install highspy"
    )

    """
    We prefer HiGHS via Pyomo's appsi interface (works with 'highspy' installed).
    """
    # 1) appsi_highs (recommended)
    try:
        s = pyo.SolverFactory("appsi_highs")
        if s.available():
            return s
    except Exception:
        pass

    # 2) highs (if you have highs executable)
    try:
        s = pyo.SolverFactory("highs")
        if s.available():
            return s
    except Exception:
        pass

    raise RuntimeError(
        "No suitable MILP solver found. Install HiGHS via: pip install highspy "
        "and ensure Pyomo can access 'appsi_highs'."
    )


def extract_solution(data, model) -> dict:
    """
    Convert key decisions into a structured JSON.
    """
    # Leader + coalition
    leaders = {}
    coalitions = {}
    for (j, i) in data.ops:
        lead = None
        team = []
        for m in data.M:
            if pyo.value(model.x[(j, i), m]) > 0.5:
                lead = m
            if pyo.value(model.y[(j, i), m]) > 0.5:
                team.append(m)
        leaders[f"{j}:{i}"] = lead
        coalitions[f"{j}:{i}"] = team

    # Operation times
    op_times = {}
    for (j, i) in data.ops:
        op_times[f"{j}:{i}"] = {
            "t": float(pyo.value(model.t[(j, i)])),
            "C": float(pyo.value(model.C[(j, i)])),
            "g": float(pyo.value(model.g[(j, i)])),
        }

    # Transfers: list used edges per stage (j,i)
    transfers = {}
    for (j, i) in data.ops:
        used = []
        for e in data.E:
            if pyo.value(model.u[(j, i), e]) > 0.5:
                used.append({
                    "edge_id": e,
                    "tail": data.tail[e],
                    "head": data.head[e],
                    "delta": data.delta[e],
                    "arm": data.edge_arm[e],
                    "host": data.edge_host[e],
                    "S": float(pyo.value(model.Sedge[(j, i), e]))
                })
        transfers[f"{j}:{i}"] = used

    # Completion at OUT
    completion = {}
    for j in data.J:
        last_i = max(data.ops_by_job[j])
        completion[j] = float(pyo.value(model.Arr[(j, last_i), data.OUT]))

    return {
        "leaders": leaders,
        "coalitions": coalitions,
        "op_times": op_times,
        "transfers": transfers,
        "completion_at_out": completion,
        "objective": float(pyo.value(model.OBJ))
    }


def write_human_readable(data, sol: dict) -> str:
    lines = []
    lines.append(f"Objective (sum completion at OUT): {sol['objective']:.3f}\n")

    lines.append("=== OPERATIONS ===")
    for (j, i) in data.ops:
        key = f"{j}:{i}"
        lead = sol["leaders"][key]
        team = sol["coalitions"][key]
        t = sol["op_times"][key]["t"]
        C = sol["op_times"][key]["C"]
        g = sol["op_times"][key]["g"]
        lines.append(f"Op {key}  leader={lead}  coalition={team}  t={t:.3f}  C={C:.3f}  g={g:.3f}")

    lines.append("\n=== TRANSFERS (used edges per stage) ===")
    for (j, i) in data.ops:
        key = f"{j}:{i}"
        edges = sol["transfers"][key]
        if not edges:
            lines.append(f"Stage {key}: (no edges selected)  <-- should not happen, check graph")
            continue
        lines.append(f"Stage {key}:")
        for e in edges:
            lines.append(
                f"  {e['edge_id']}: {e['tail']} -> {e['head']}  "
                f"delta={e['delta']}  arm={e['arm']} host={e['host']}  S={e['S']:.3f}"
            )

    lines.append("\n=== COMPLETION AT OUT ===")
    for j in data.J:
        lines.append(f"Job {j}: A[last,OUT] = {sol['completion_at_out'][j]:.3f}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to JSON instance, e.g. test.json")
    ap.add_argument("--out", type=str, default="outputs", help="Output folder")
    args = ap.parse_args()

    data = load_instance(args.data)
    model = build_model(data)

    solver = pick_solver()
    result = solver.solve(model, tee=True)

    # Basic solve status check
    term = str(result.solver.termination_condition).lower()
    if "optimal" not in term and "feasible" not in term:
        print("WARNING: Solver termination condition:", result.solver.termination_condition)

    sol = extract_solution(data, model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "solution.json").write_text(json.dumps(sol, indent=2), encoding="utf-8")
    (out_dir / "solution.txt").write_text(write_human_readable(data, sol), encoding="utf-8")

    print("\nWrote:")
    print(" -", out_dir / "solution.json")
    print(" -", out_dir / "solution.txt")


if __name__ == "__main__":
    main()
