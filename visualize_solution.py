from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass
class Interval:
    start: float
    end: float
    label: str
    job: str
    kind: str


def parse_stage_key(stage_key: str) -> tuple[str, int]:
    job, op_index = stage_key.split(":")
    return job, int(op_index)


def order_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not edges:
        return []

    out_map: dict[str, dict[str, Any]] = {}
    tails: set[str] = set()
    heads: set[str] = set()
    for edge in edges:
        out_map[edge["tail"]] = edge
        tails.add(edge["tail"])
        heads.add(edge["head"])

    sources = list(tails - heads)
    if len(sources) != 1:
        return edges

    current = sources[0]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    while current in out_map:
        edge = out_map[current]
        edge_id = str(edge.get("edge_id", id(edge)))
        if edge_id in seen:
            break
        seen.add(edge_id)
        ordered.append(edge)
        current = edge["head"]
    return ordered


def load_solution(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def edge_arms(edge: dict[str, Any]) -> list[str]:
    return as_list(edge.get("arms", edge.get("arm")))


def edge_hosts(edge: dict[str, Any]) -> list[str]:
    return as_list(edge.get("hosts", edge.get("host")))


def infer_resources(solution: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    transfers = solution.get("transfers", {})

    machines: set[str] = set()
    arms: set[str] = set()
    nodes: set[str] = set()
    indegree: dict[str, int] = defaultdict(int)
    outdegree: dict[str, int] = defaultdict(int)

    for leader in leaders.values():
        if leader:
            machines.add(leader)

    for team in coalitions.values():
        for machine in team:
            machines.add(machine)

    for edges in transfers.values():
        for edge in edges:
            tail = edge.get("tail")
            head = edge.get("head")
            for arm in edge_arms(edge):
                arms.add(arm)
            for host in edge_hosts(edge):
                machines.add(host)
            if tail:
                nodes.add(tail)
                outdegree[str(tail)] += 1
            if head:
                nodes.add(head)
                indegree[str(head)] += 1

    entry_nodes = {
        node
        for node in nodes
        if node not in machines and node != "OUT" and outdegree[node] > 0 and indegree[node] == 0
    }
    buffers = sorted(
        node for node in nodes if node not in machines and node != "OUT" and node not in entry_nodes
    )
    return sorted(machines), sorted(arms), buffers


def build_intervals(
    solution: dict[str, Any],
    machines: list[str],
    arms: list[str],
    buffers: list[str],
) -> tuple[dict[str, list[Interval]], dict[str, list[Interval]], dict[str, list[Interval]], float]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    op_times = solution.get("op_times", {})
    transfers = solution.get("transfers", {})

    machine_intervals: dict[str, list[Interval]] = {machine: [] for machine in machines}
    arm_intervals: dict[str, list[Interval]] = {arm: [] for arm in arms}
    buffer_intervals: dict[str, list[Interval]] = {buffer: [] for buffer in buffers}
    horizon = 0.0

    for stage_key, timing in op_times.items():
        job, _ = parse_stage_key(stage_key)
        start = float(timing["t"])
        completion = float(timing["C"])
        block_end = float(timing.get("g", completion))
        leader = leaders.get(stage_key)
        team = coalitions.get(stage_key, [])

        for machine in team:
            if machine in machine_intervals:
                machine_intervals[machine].append(
                    Interval(start, completion, f"{stage_key} proc", job, "proc")
                )
                horizon = max(horizon, completion)

        if leader in machine_intervals and block_end > completion + 1e-9:
            machine_intervals[leader].append(
                Interval(completion, block_end, f"{stage_key} block", job, "block")
            )
            horizon = max(horizon, block_end)

    for stage_key, edges in transfers.items():
        job, _ = parse_stage_key(stage_key)
        ordered = order_edges(edges)
        if not ordered:
            continue

        for edge in ordered:
            start = float(edge["S"])
            end = start + float(edge["delta"])
            short_label = f"{stage_key} {edge.get('tail')}->{edge.get('head')}"

            for arm in edge_arms(edge):
                if arm in arm_intervals:
                    arm_intervals[arm].append(Interval(start, end, short_label, job, "transfer"))
            for host in edge_hosts(edge):
                if host in machine_intervals:
                    machine_intervals[host].append(
                        Interval(start, end, f"{stage_key} arm-use", job, "host-transfer")
                    )
            horizon = max(horizon, end)

        for index in range(1, len(ordered)):
            prev_edge = ordered[index - 1]
            next_edge = ordered[index]
            node = prev_edge["head"]
            if node not in buffer_intervals:
                continue

            arrival = float(prev_edge["S"]) + float(prev_edge["delta"])
            departure = float(next_edge["S"])
            if departure > arrival + 1e-9:
                buffer_intervals[node].append(
                    Interval(arrival, departure, f"{stage_key} wait", job, "buffer")
                )
                horizon = max(horizon, departure)

    return machine_intervals, arm_intervals, buffer_intervals, max(horizon, 1.0)


def build_stage_summary(solution: dict[str, Any]) -> list[str]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    transfers = solution.get("transfers", {})

    stage_keys = sorted(
        set(leaders) | set(coalitions) | set(transfers),
        key=lambda item: parse_stage_key(item),
    )

    lines: list[str] = []
    for stage_key in stage_keys:
        leader = leaders.get(stage_key, "-")
        team = ",".join(coalitions.get(stage_key, [])) or "-"
        edges = order_edges(transfers.get(stage_key, []))
        if edges:
            route = " -> ".join([edges[0]["tail"], *[edge["head"] for edge in edges]])
        else:
            route = "stay"
        lines.append(f"{stage_key}: leader={leader} team=[{team}] route={route}")
    return lines


def job_colors(solution: dict[str, Any]) -> dict[str, Any]:
    stage_keys = sorted(solution.get("op_times", {}), key=lambda item: parse_stage_key(item))
    jobs = []
    for stage_key in stage_keys:
        job, _ = parse_stage_key(stage_key)
        if job not in jobs:
            jobs.append(job)

    cmap = plt.get_cmap("tab10")
    return {job: cmap(index % cmap.N) for index, job in enumerate(jobs)}


def draw_interval(
    ax: Any,
    row: int,
    interval: Interval,
    color: Any,
    horizon: float,
) -> None:
    width = max(0.0, interval.end - interval.start)
    if width <= 1e-9:
        return

    style = {
        "proc": {"alpha": 0.85, "hatch": None, "edgecolor": "black"},
        "block": {"alpha": 0.35, "hatch": "//", "edgecolor": "black"},
        "host-transfer": {"alpha": 0.45, "hatch": "xx", "edgecolor": "black"},
        "transfer": {"alpha": 0.70, "hatch": None, "edgecolor": "black"},
        "buffer": {"alpha": 0.50, "hatch": "..", "edgecolor": "black"},
    }[interval.kind]

    ax.barh(
        row,
        width,
        left=interval.start,
        height=0.8,
        color=color,
        alpha=style["alpha"],
        hatch=style["hatch"],
        edgecolor=style["edgecolor"],
        linewidth=0.6,
    )

    if width >= horizon * 0.08:
        ax.text(
            interval.start + width / 2,
            row,
            interval.label,
            ha="center",
            va="center",
            fontsize=7,
        )


def render_solution_visualization(solution: dict[str, Any], out_path: Path) -> Path:
    machines, arms, buffers = infer_resources(solution)
    machine_intervals, arm_intervals, buffer_intervals, horizon = build_intervals(
        solution, machines, arms, buffers
    )
    colors = job_colors(solution)

    rows: list[tuple[str, str]] = []
    rows.extend(("M", machine) for machine in machines)
    rows.extend(("A", arm) for arm in arms)
    rows.extend(("B", buffer) for buffer in buffers)
    row_map = {row: index for index, row in enumerate(rows)}

    completions = solution.get("completion_at_out", {})
    sorted_jobs = sorted(completions)
    stage_summary = build_stage_summary(solution)

    figure_height = max(8.0, 0.45 * len(rows) + 4.5)
    fig = plt.figure(figsize=(15, figure_height))
    axes = fig.subplot_mosaic(
        [["gantt", "gantt"], ["completion", "summary"]],
        height_ratios=[3.2, 1.4],
        width_ratios=[1.1, 1.3],
    )

    gantt_ax = axes["gantt"]
    for machine in machines:
        row = row_map[("M", machine)]
        for interval in sorted(machine_intervals[machine], key=lambda item: item.start):
            draw_interval(gantt_ax, row, interval, colors[interval.job], horizon)

    for arm in arms:
        row = row_map[("A", arm)]
        for interval in sorted(arm_intervals[arm], key=lambda item: item.start):
            draw_interval(gantt_ax, row, interval, colors[interval.job], horizon)

    for buffer in buffers:
        row = row_map[("B", buffer)]
        for interval in sorted(buffer_intervals[buffer], key=lambda item: item.start):
            draw_interval(gantt_ax, row, interval, colors[interval.job], horizon)

    y_ticks: list[int] = []
    y_labels: list[str] = []
    for (kind, name), row in row_map.items():
        y_ticks.append(row)
        if kind == "M":
            y_labels.append(f"Machine {name}")
        elif kind == "A":
            y_labels.append(f"Arm {name}")
        else:
            y_labels.append(f"Buffer {name}")

    gantt_ax.set_yticks(y_ticks)
    gantt_ax.set_yticklabels(y_labels)
    gantt_ax.invert_yaxis()
    gantt_ax.set_xlim(0, horizon * 1.05)
    gantt_ax.set_xlabel("Time")
    gantt_ax.set_title("Schedule Overview From solution.json")
    gantt_ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.85, label="processing"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.35, hatch="//", label="leader block"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.45, hatch="xx", label="host busy by arm"),
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.70, label="arm transfer"),
    ]
    if buffers:
        legend_handles.append(
            Patch(facecolor="lightgray", edgecolor="black", alpha=0.50, hatch="..", label="buffer wait")
        )
    gantt_ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    completion_ax = axes["completion"]
    completion_values = [float(completions[job]) for job in sorted_jobs]
    completion_colors = [colors.get(job, "tab:blue") for job in sorted_jobs]
    completion_ax.barh(sorted_jobs, completion_values, color=completion_colors, edgecolor="black")
    completion_ax.set_title("Completion At OUT")
    completion_ax.set_xlabel("Time")
    completion_ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    objective = float(solution.get("objective", 0.0))
    completion_ax.text(
        0.98,
        0.05,
        f"Objective: {objective:.2f}",
        transform=completion_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7"},
    )

    summary_ax = axes["summary"]
    summary_ax.axis("off")
    summary_text = "\n".join(stage_summary)
    summary_ax.set_title("Stage Summary", loc="left")
    summary_ax.text(
        0.0,
        1.0,
        summary_text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.5,
        wrap=True,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a solver solution JSON as a standalone PNG.")
    parser.add_argument(
        "solution",
        nargs="?",
        default="outputs/solution.json",
        help="Path to solution.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (defaults to solution_visualization.png next to the JSON file)",
    )
    args = parser.parse_args()

    solution_path = Path(args.solution)
    out_path = (
        Path(args.out)
        if args.out is not None
        else solution_path.with_name("solution_visualization.png")
    )

    solution = load_solution(solution_path)
    rendered = render_solution_visualization(solution, out_path)
    print(f"Wrote: {rendered}")


if __name__ == "__main__":
    main()
