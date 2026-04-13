# src/gantt.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def _get(data: Any, key: str, default=None):
    """Support both dict-style and attribute-style access."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


def _parse_stage_key(stage_key: str) -> Tuple[str, int]:
    """'J1:2' -> ('J1', 2)"""
    j, i = stage_key.split(":")
    return j, int(i)


def _order_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order used edges of a stage into a path tail->...->head if possible.
    If ambiguous or empty, return as-is.
    """
    if not edges:
        return []

    # Build: tail -> edge (assumes path, no splitting)
    out_map = {}
    tails = set()
    heads = set()
    for e in edges:
        out_map[e["tail"]] = e
        tails.add(e["tail"])
        heads.add(e["head"])

    # Source = tail that is not a head
    sources = list(tails - heads)
    if len(sources) != 1:
        return edges  # cannot uniquely order

    cur = sources[0]
    ordered = []
    seen = set()
    while cur in out_map:
        e = out_map[cur]
        eid = e.get("edge_id", id(e))
        if eid in seen:
            break
        seen.add(eid)
        ordered.append(e)
        cur = e["head"]
    return ordered


def render_gantt(data: Any, solution: Dict[str, Any], out_dir: Path) -> Path:
    """
    Creates a Gantt chart PNG into out_dir.
    Shows machine processing, optional leader blocking, arm transfers, and host-occupancy during transfers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    leaders: Dict[str, str] = solution.get("leaders", {})
    coalitions: Dict[str, List[str]] = solution.get("coalitions", {})
    op_times: Dict[str, Dict[str, float]] = solution.get("op_times", {})
    transfers: Dict[str, List[Dict[str, Any]]] = solution.get("transfers", {})

    # Resource sets
    machines = set(_get(data, "machines", []) or [])
    arms = set(_get(data, "arms", []) or [])
    buffers = set(_get(data, "buffers", []) or [])
    entries = set(_get(data, "V_in", []) or [])

    # Fallback: infer from solution if not in data
    for op, leader in leaders.items():
        machines.add(leader)
    for op, coal in coalitions.items():
        for m in coal:
            machines.add(m)
    for stage_key, edges in transfers.items():
        for e in edges:
            if "arm" in e:
                arms.add(e["arm"])
            if "host" in e:
                machines.add(e["host"])

    # Infer buffers/entries from routing if not in machines and not OUT
    for stage_key, edges in transfers.items():
        for e in edges:
            for node in (e["tail"], e["head"]):
                if node not in machines and node != "OUT":
                    buffers.add(node)

    machines = sorted(machines)
    arms = sorted(arms)
    buffers = sorted(buffers)

    # Build interval lists per resource row
    # Each entry: (start, end, label)
    machine_intervals: Dict[str, List[Tuple[float, float, str]]] = {m: [] for m in machines}
    arm_intervals: Dict[str, List[Tuple[float, float, str]]] = {a: [] for a in arms}
    buffer_intervals: Dict[str, List[Tuple[float, float, str]]] = {b: [] for b in buffers}

    # --- Machine processing + leader blocking ---
    for op_key, times in op_times.items():
        t = float(times["t"])
        C = float(times["C"])
        g = float(times.get("g", C))

        leader = leaders.get(op_key)
        coal = coalitions.get(op_key, [])

        # Processing occupies every coalition machine in [t, C]
        for m in coal:
            if m in machine_intervals:
                tag = f"{op_key} proc"
                machine_intervals[m].append((t, C, tag))

        # Leader blocking (only if >0)
        if leader is not None and leader in machine_intervals and g > C + 1e-9:
            machine_intervals[leader].append((C, g, f"{op_key} block"))

    # --- Arm transfers + host machine occupied during transfer ---
    # Also reconstruct buffer occupancy if buffers exist and edges can be ordered
    for stage_key, edges in transfers.items():
        edges_ord = _order_edges(edges)
        if not edges_ord:
            # could be "stay" (no transport needed)
            continue

        # Arm + host occupancy per edge
        for e in edges_ord:
            S = float(e["S"])
            delta = float(e["delta"])
            end = S + delta

            arm = e.get("arm")
            host = e.get("host")
            eid = e.get("edge_id", "edge")

            # Arm busy on [S, S+delta]
            if arm in arm_intervals:
                arm_intervals[arm].append((S, end, f"{stage_key} {eid}"))

            # Host machine busy during transfer on [S, S+delta]
            if host in machine_intervals:
                machine_intervals[host].append((S, end, f"{stage_key} arm-use"))

        # Buffer occupancy: for a buffer node b visited, occupied from arrival to departure
        # arrival at node = (prev S + prev delta), departure from node = (next S)
        # Only possible if buffers list exists.
        if buffer_intervals:
            for idx in range(1, len(edges_ord)):
                prev_e = edges_ord[idx - 1]
                next_e = edges_ord[idx]
                node = prev_e["head"]  # intermediate node reached after prev edge

                if node in buffer_intervals:
                    arr = float(prev_e["S"]) + float(prev_e["delta"])
                    dep = float(next_e["S"])
                    if dep > arr + 1e-9:
                        buffer_intervals[node].append((arr, dep, f"{stage_key} buf"))

    # --- Build figure rows ---
    # We show Machines first, then Arms, then Buffers
    rows: List[Tuple[str, str]] = []  # (kind, name)
    for m in machines:
        rows.append(("M", m))
    for a in arms:
        rows.append(("A", a))
    for b in buffers:
        rows.append(("B", b))

    # Determine plot horizon
    max_t = 0.0
    for m, lst in machine_intervals.items():
        for s, e, _ in lst:
            max_t = max(max_t, e)
    for a, lst in arm_intervals.items():
        for s, e, _ in lst:
            max_t = max(max_t, e)
    for b, lst in buffer_intervals.items():
        for s, e, _ in lst:
            max_t = max(max_t, e)

    if max_t <= 0:
        max_t = 1.0

    # --- Plot ---
    fig_h = max(5.0, 0.35 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y_map = {row: i for i, row in enumerate(rows)}
    bar_h = 0.8

    def add_bar(y: int, start: float, end: float, label: str):
        width = max(0.0, end - start)
        if width <= 1e-9:
            return
        ax.barh(y, width, left=start, height=bar_h)  # default colors
        # label centered (small font)
        ax.text(start + width / 2, y, label, ha="center", va="center", fontsize=7)

    # Machines
    for m in machines:
        y = y_map[("M", m)]
        for s, e, lab in sorted(machine_intervals[m], key=lambda x: x[0]):
            add_bar(y, s, e, lab)

    # Arms
    for a in arms:
        y = y_map[("A", a)]
        for s, e, lab in sorted(arm_intervals[a], key=lambda x: x[0]):
            add_bar(y, s, e, lab)

    # Buffers
    for b in buffers:
        y = y_map[("B", b)]
        for s, e, lab in sorted(buffer_intervals[b], key=lambda x: x[0]):
            add_bar(y, s, e, lab)

    # y-axis labels
    yticks = []
    ylabels = []
    for (kind, name), y in y_map.items():
        yticks.append(y)
        if kind == "M":
            ylabels.append(f"Machine {name}")
        elif kind == "A":
            ylabels.append(f"Arm {name}")
        else:
            if name in entries:
                ylabels.append(f"Entry {name}")
            else:
                ylabels.append(f"Buffer {name}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_xlim(0, max_t * 1.05)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
    ax.invert_yaxis()  # top row = first machine

    ax.set_title("Gantt chart: Machines / Arms / Buffers")

    out_path = out_dir / "gantt.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path
