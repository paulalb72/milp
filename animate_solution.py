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
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle


@dataclass
class StageInfo:
    stage_key: str
    job: str
    op_index: int
    leader: str
    coalition: list[str]
    start: float
    completion: float
    stage_end: float
    edges: list[dict[str, Any]]


def parse_stage_key(stage_key: str) -> tuple[str, int]:
    job, op_index = stage_key.split(":")
    return job, int(op_index)


def load_solution(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def infer_resources(solution: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    transfers = solution.get("transfers", {})

    machines: set[str] = set()
    nodes: set[str] = {"OUT"}
    used_edges: dict[str, dict[str, Any]] = {}

    for leader in leaders.values():
        if leader:
            machines.add(leader)
            nodes.add(leader)

    for team in coalitions.values():
        for machine in team:
            machines.add(machine)
            nodes.add(machine)

    for edges in transfers.values():
        for edge in edges:
            edge_id = edge["edge_id"]
            used_edges[edge_id] = edge
            nodes.add(edge["tail"])
            nodes.add(edge["head"])
            host = edge.get("host")
            if host:
                machines.add(host)
                nodes.add(host)

    buffers = sorted(node for node in nodes if node not in machines and node != "OUT")
    return sorted(machines), buffers, ["OUT"], list(used_edges.values())


def build_stage_infos(solution: dict[str, Any]) -> tuple[dict[str, list[StageInfo]], float]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    op_times = solution.get("op_times", {})
    transfers = solution.get("transfers", {})
    completion_at_out = solution.get("completion_at_out", {})

    stages_by_job: dict[str, list[StageInfo]] = defaultdict(list)
    for stage_key, timing in op_times.items():
        job, op_index = parse_stage_key(stage_key)
        stages_by_job[job].append(
            StageInfo(
                stage_key=stage_key,
                job=job,
                op_index=op_index,
                leader=leaders[stage_key],
                coalition=list(coalitions.get(stage_key, [])),
                start=float(timing["t"]),
                completion=float(timing["C"]),
                stage_end=0.0,
                edges=order_edges(transfers.get(stage_key, [])),
            )
        )

    horizon = 0.0
    for job, stages in stages_by_job.items():
        stages.sort(key=lambda item: item.op_index)
        for index, stage in enumerate(stages):
            if index + 1 < len(stages):
                stage.stage_end = stages[index + 1].start
            else:
                stage.stage_end = float(completion_at_out[job])
            horizon = max(horizon, stage.stage_end)

    return dict(stages_by_job), max(horizon, 1.0)


def evenly_spaced_positions(count: int) -> list[float]:
    if count <= 1:
        return [0.5]
    return [0.85 - index * (0.70 / (count - 1)) for index in range(count)]


def layout_nodes(machines: list[str], buffers: list[str], out_nodes: list[str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}

    for machine, y in zip(machines, evenly_spaced_positions(len(machines))):
        positions[machine] = (0.16, y)

    for buffer_name, y in zip(buffers, evenly_spaced_positions(len(buffers))):
        positions[buffer_name] = (0.52, y)

    for out_name, y in zip(out_nodes, evenly_spaced_positions(len(out_nodes))):
        positions[out_name] = (0.86, y)

    return positions


def job_colors(stages_by_job: dict[str, list[StageInfo]]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    jobs = sorted(stages_by_job)
    return {job: cmap(index % cmap.N) for index, job in enumerate(jobs)}


def lerp(start: tuple[float, float], end: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * alpha,
        start[1] + (end[1] - start[1]) * alpha,
    )


def job_state_at(
    stages_by_job: dict[str, list[StageInfo]],
    job: str,
    current_time: float,
    positions: dict[str, tuple[float, float]],
    horizon: float,
) -> dict[str, Any]:
    stages = stages_by_job[job]
    first_stage = stages[0]
    if current_time < first_stage.start:
        return {
            "position": positions[first_stage.leader],
            "state": f"waiting for release on {first_stage.leader}",
            "group": first_stage.leader,
            "node": first_stage.leader,
            "edge_id": None,
            "active_stage": None,
            "active_coalition": [],
            "machine_roles": [],
        }

    for stage in stages:
        if current_time < stage.start:
            continue
        if current_time > stage.stage_end + 1e-9:
            continue

        if current_time < stage.completion:
            return {
                "position": positions[stage.leader],
                "state": f"{stage.stage_key} processing on {stage.leader}",
                "group": stage.leader,
                "node": stage.leader,
                "edge_id": None,
                "active_stage": stage.stage_key,
                "active_coalition": stage.coalition,
                "machine_roles": [(machine, "processing") for machine in stage.coalition],
            }

        if not stage.edges:
            state = "waiting for next op"
            if current_time >= stage.stage_end - 1e-9 and stage.stage_end >= horizon - 1e-9:
                state = "completed"
            return {
                "position": positions[stage.leader],
                "state": f"{stage.stage_key} {state} on {stage.leader}",
                "group": stage.leader,
                "node": stage.leader,
                "edge_id": None,
                "active_stage": stage.stage_key,
                "active_coalition": [],
                "machine_roles": [],
            }

        first_edge = stage.edges[0]
        if current_time < float(first_edge["S"]):
            return {
                "position": positions[stage.leader],
                "state": f"{stage.stage_key} waiting for transfer on {stage.leader}",
                "group": stage.leader,
                "node": stage.leader,
                "edge_id": None,
                "active_stage": stage.stage_key,
                "active_coalition": [],
                "machine_roles": [],
            }

        for index, edge in enumerate(stage.edges):
            edge_start = float(edge["S"])
            edge_end = edge_start + float(edge["delta"])
            tail_pos = positions[edge["tail"]]
            head_pos = positions[edge["head"]]

            if edge_start <= current_time < edge_end:
                alpha = (current_time - edge_start) / max(float(edge["delta"]), 1e-9)
                return {
                    "position": lerp(tail_pos, head_pos, alpha),
                    "state": f"{stage.stage_key} transfer {edge['tail']} -> {edge['head']}",
                    "group": edge["edge_id"],
                    "node": None,
                    "edge_id": edge["edge_id"],
                    "active_stage": stage.stage_key,
                    "active_coalition": [],
                    "machine_roles": [(edge["host"], "arm-use")] if edge.get("host") else [],
                }

            next_start = stage.stage_end
            if index + 1 < len(stage.edges):
                next_start = float(stage.edges[index + 1]["S"])

            if edge_end <= current_time < next_start:
                return {
                    "position": positions[edge["head"]],
                    "state": f"{stage.stage_key} waiting on {edge['head']}",
                    "group": edge["head"],
                    "node": edge["head"],
                    "edge_id": None,
                    "active_stage": stage.stage_key,
                    "active_coalition": [],
                    "machine_roles": [],
                }

        last_head = stage.edges[-1]["head"]
        final_state = "completed" if last_head == "OUT" else f"waiting on {last_head}"
        return {
            "position": positions[last_head],
            "state": f"{stage.stage_key} {final_state}",
            "group": last_head,
            "node": last_head,
            "edge_id": None,
            "active_stage": stage.stage_key,
            "active_coalition": [],
            "machine_roles": [],
        }

    return {
        "position": positions["OUT"],
        "state": "completed",
        "group": "OUT",
        "node": "OUT",
        "edge_id": None,
        "active_stage": None,
        "active_coalition": [],
        "machine_roles": [],
    }


def distribute_overlaps(states: dict[str, dict[str, Any]]) -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for job, state in states.items():
        groups[str(state["group"])].append(job)

    for jobs in groups.values():
        if len(jobs) == 1:
            continue
        for index, job in enumerate(sorted(jobs)):
            offset = (index - (len(jobs) - 1) / 2) * 0.022
            x, y = states[job]["position"]
            states[job]["position"] = (x, y + offset)


def collect_machine_activity(states: dict[str, dict[str, Any]], machines: list[str]) -> dict[str, list[tuple[str, str]]]:
    activity: dict[str, list[tuple[str, str]]] = {machine: [] for machine in machines}
    for job, state in states.items():
        for machine, role in state.get("machine_roles", []):
            if machine in activity:
                activity[machine].append((job, role))
    return activity


def build_timeline_axes(
    timeline_ax: Any,
    stages_by_job: dict[str, list[StageInfo]],
    colors: dict[str, Any],
    horizon: float,
) -> tuple[Line2D, list[Patch]]:
    jobs = sorted(stages_by_job)
    row_map = {job: index for index, job in enumerate(jobs)}

    for job in jobs:
        y = row_map[job]
        for stage in stages_by_job[job]:
            timeline_ax.barh(
                y + 0.12,
                stage.completion - stage.start,
                left=stage.start,
                height=0.22,
                color=colors[job],
                edgecolor="black",
                alpha=0.80,
            )
            for edge in stage.edges:
                edge_start = float(edge["S"])
                edge_width = float(edge["delta"])
                timeline_ax.barh(
                    y - 0.15,
                    edge_width,
                    left=edge_start,
                    height=0.18,
                    color=colors[job],
                    edgecolor="black",
                    alpha=0.55,
                    hatch="//",
                )

    timeline_ax.set_xlim(0, horizon * 1.02)
    timeline_ax.set_yticks([row_map[job] for job in jobs])
    timeline_ax.set_yticklabels(jobs)
    timeline_ax.set_xlabel("Time")
    timeline_ax.set_title("Processing And Transfer Timeline")
    timeline_ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    time_cursor = timeline_ax.axvline(0.0, color="crimson", linewidth=2.0)
    legend = [
        Patch(facecolor="gray", edgecolor="black", alpha=0.80, label="processing"),
        Patch(facecolor="gray", edgecolor="black", alpha=0.55, hatch="//", label="transfer"),
    ]
    return time_cursor, legend


def render_animation(solution: dict[str, Any], out_path: Path, fps: int, time_step: float) -> Path:
    stages_by_job, horizon = build_stage_infos(solution)
    machines, buffers, out_nodes, used_edges = infer_resources(solution)
    positions = layout_nodes(machines, buffers, out_nodes)
    colors = job_colors(stages_by_job)

    fig = plt.figure(figsize=(14, 8.5))
    network_ax = fig.add_axes([0.05, 0.28, 0.90, 0.67])
    timeline_ax = fig.add_axes([0.08, 0.08, 0.84, 0.14])

    network_ax.set_xlim(0.0, 1.0)
    network_ax.set_ylim(0.0, 1.0)
    network_ax.set_aspect("equal")
    network_ax.axis("off")
    network_ax.set_title("Animated Solution Flow")

    edge_artists: dict[str, FancyArrowPatch] = {}
    edge_labels: list[Any] = []
    for edge in used_edges:
        start = positions[edge["tail"]]
        end = positions[edge["head"]]
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color="0.75",
            alpha=0.6,
            zorder=1,
        )
        network_ax.add_patch(arrow)
        edge_artists[edge["edge_id"]] = arrow

        label_pos = lerp(start, end, 0.5)
        edge_labels.append(
            network_ax.text(
                label_pos[0],
                label_pos[1] + 0.03,
                edge["edge_id"],
                ha="center",
                va="center",
                fontsize=7,
                color="0.35",
            )
        )

    node_patches: dict[str, Any] = {}
    for machine in machines:
        x, y = positions[machine]
        patch = Circle((x, y), 0.055, facecolor="white", edgecolor="black", linewidth=1.8, zorder=3)
        network_ax.add_patch(patch)
        node_patches[machine] = patch
        network_ax.text(x, y, machine, ha="center", va="center", fontsize=10, weight="bold", zorder=4)

    for buffer_name in buffers:
        x, y = positions[buffer_name]
        patch = Rectangle((x - 0.06, y - 0.04), 0.12, 0.08, facecolor="white", edgecolor="black", linewidth=1.8, zorder=3)
        network_ax.add_patch(patch)
        node_patches[buffer_name] = patch
        network_ax.text(x, y, buffer_name, ha="center", va="center", fontsize=10, weight="bold", zorder=4)

    out_x, out_y = positions["OUT"]
    out_patch = Circle((out_x, out_y), 0.065, facecolor="#eef7e8", edgecolor="black", linewidth=2.0, zorder=3)
    network_ax.add_patch(out_patch)
    node_patches["OUT"] = out_patch
    network_ax.text(out_x, out_y, "OUT", ha="center", va="center", fontsize=11, weight="bold", zorder=4)

    machine_status_texts: dict[str, Any] = {}
    for machine in machines:
        x, y = positions[machine]
        machine_status_texts[machine] = network_ax.text(
            x + 0.08,
            y,
            "",
            ha="left",
            va="center",
            fontsize=8,
            color="0.25",
            zorder=4,
        )

    job_tokens: dict[str, Any] = {}
    job_labels: dict[str, Any] = {}
    for job, color in colors.items():
        token, = network_ax.plot([], [], marker="o", markersize=12, color=color, markeredgecolor="black", zorder=6)
        label = network_ax.text(0.0, 0.0, job, ha="left", va="bottom", fontsize=9, weight="bold", color=color, zorder=7)
        job_tokens[job] = token
        job_labels[job] = label

    time_text = network_ax.text(
        0.01,
        0.98,
        "",
        transform=network_ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.7"},
    )
    status_text = network_ax.text(
        0.99,
        0.98,
        "",
        transform=network_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.7"},
    )

    time_cursor, timeline_legend = build_timeline_axes(timeline_ax, stages_by_job, colors, horizon)
    timeline_ax.legend(handles=timeline_legend, loc="upper right", fontsize=8)

    job_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[job], markeredgecolor="black", label=job, markersize=9) for job in sorted(colors)]
    network_ax.legend(handles=job_legend, loc="lower left", fontsize=8, title="Jobs")

    frame_count = int(horizon / time_step) + 2

    def update(frame_index: int) -> list[Any]:
        current_time = min(frame_index * time_step, horizon)
        states = {
            job: job_state_at(stages_by_job, job, current_time, positions, horizon)
            for job in sorted(stages_by_job)
        }
        distribute_overlaps(states)
        machine_activity = collect_machine_activity(states, machines)

        for patch in edge_artists.values():
            patch.set_color("0.75")
            patch.set_alpha(0.6)
            patch.set_linewidth(1.8)

        for name, patch in node_patches.items():
            if name == "OUT":
                patch.set_facecolor("#eef7e8")
            else:
                patch.set_facecolor("white")
            patch.set_alpha(1.0)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.8)

        active_edges: dict[str, list[str]] = defaultdict(list)
        status_lines = []
        for job, state in states.items():
            x, y = state["position"]
            job_tokens[job].set_data([x], [y])
            job_labels[job].set_position((x + 0.015, y + 0.015))
            job_labels[job].set_text(job)
            status_lines.append(f"{job:<3} | {state['state']}")

            if state["edge_id"]:
                active_edges[state["edge_id"]].append(job)

        for edge_id, jobs in active_edges.items():
            lead_job = sorted(jobs)[0]
            edge_artists[edge_id].set_color(colors[lead_job])
            edge_artists[edge_id].set_alpha(0.95)
            edge_artists[edge_id].set_linewidth(3.0)

        for machine, activities in machine_activity.items():
            if not activities:
                machine_status_texts[machine].set_text("")
                continue

            lead_job, lead_role = activities[0]
            node_patches[machine].set_facecolor(colors[lead_job])
            node_patches[machine].set_alpha(0.30)
            if lead_role == "arm-use":
                node_patches[machine].set_edgecolor("darkorange")
                node_patches[machine].set_linewidth(2.6)
            else:
                node_patches[machine].set_edgecolor(colors[lead_job])
                node_patches[machine].set_linewidth(2.6)
            machine_status_texts[machine].set_text(", ".join(f"{job}:{role}" for job, role in activities))

        for node_name in buffers + out_nodes:
            occupants = [job for job, state in states.items() if state["node"] == node_name]
            if occupants and node_name in node_patches:
                lead_job = sorted(occupants)[0]
                node_patches[node_name].set_facecolor(colors[lead_job])
                node_patches[node_name].set_alpha(0.25 if node_name != "OUT" else 0.35)

        time_text.set_text(f"time = {current_time:.2f}")
        status_text.set_text("\n".join(status_lines))
        time_cursor.set_xdata([current_time, current_time])
        return [
            *edge_artists.values(),
            *node_patches.values(),
            *job_tokens.values(),
            *job_labels.values(),
            *machine_status_texts.values(),
            time_text,
            status_text,
            time_cursor,
        ]

    anim = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False, repeat=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a solver solution JSON as a GIF.")
    parser.add_argument(
        "solution",
        nargs="?",
        default="outputs/solution.json",
        help="Path to solution.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output GIF path (defaults to solution_animation.gif next to the JSON file)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for the GIF",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.25,
        help="Simulated time units between animation frames",
    )
    args = parser.parse_args()

    solution_path = Path(args.solution)
    out_path = (
        Path(args.out)
        if args.out is not None
        else solution_path.with_name("solution_animation.gif")
    )

    solution = load_solution(solution_path)
    rendered = render_animation(solution, out_path, fps=args.fps, time_step=args.time_step)
    print(f"Wrote: {rendered}")


if __name__ == "__main__":
    main()
