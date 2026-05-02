from __future__ import annotations

import argparse
import json
import math
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


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def edge_hosts(edge: dict[str, Any]) -> list[str]:
    return as_list(edge.get("hosts", edge.get("host")))


def load_instance_graph(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sets = raw.get("sets", {})
    arm_hosts = {
        str(arm_id): str(info.get("host_machine"))
        for arm_id, info in raw.get("arms", {}).items()
    }

    edges: list[dict[str, Any]] = []
    for edge in raw.get("graph", {}).get("edges", []):
        arms = as_list(edge.get("served_by_arms", edge.get("served_by_arm")))
        hosts = unique_preserve_order([arm_hosts[arm] for arm in arms if arm in arm_hosts])
        edges.append(
            {
                "edge_id": str(edge["edge_id"]),
                "tail": str(edge["tail"]),
                "head": str(edge["head"]),
                "delta": float(edge.get("delta", 0.0)),
                "arms": arms,
                "hosts": hosts,
            }
        )

    return {
        "machines": [str(item) for item in sets.get("M", [])],
        "entry_nodes": [str(item) for item in sets.get("V_in", [])],
        "buffers": [str(item) for item in sets.get("B", [])],
        "out_nodes": [str(sets.get("V_out", "OUT"))],
        "edges": edges,
    }


def default_data_path_for_solution(solution_path: Path) -> Path | None:
    parts = solution_path.parts
    if len(parts) >= 3 and parts[-1] == "solution.json" and parts[-3] == "outputs":
        candidate = Path(f"{parts[-2]}.json")
        if candidate.exists():
            return candidate
    return None


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


def infer_resources(
    solution: dict[str, Any],
    instance_graph: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str], list[str], list[dict[str, Any]]]:
    leaders = solution.get("leaders", {})
    coalitions = solution.get("coalitions", {})
    transfers = solution.get("transfers", {})

    machines: set[str] = set()
    nodes: set[str] = {"OUT"}
    used_edges: dict[str, dict[str, Any]] = {}
    graph_edges: dict[str, dict[str, Any]] = {}
    indegree: dict[str, int] = defaultdict(int)
    outdegree: dict[str, int] = defaultdict(int)

    entry_order: list[str] = []
    buffer_order: list[str] = []
    out_order: list[str] = ["OUT"]
    machine_order: list[str] = []

    if instance_graph is not None:
        machine_order.extend(instance_graph.get("machines", []))
        entry_order.extend(instance_graph.get("entry_nodes", []))
        buffer_order.extend(instance_graph.get("buffers", []))
        out_order = list(instance_graph.get("out_nodes", ["OUT"]))
        machines.update(machine_order)
        nodes.update(machine_order)
        nodes.update(entry_order)
        nodes.update(buffer_order)
        nodes.update(out_order)
        for edge in instance_graph.get("edges", []):
            graph_edges[edge["edge_id"]] = edge

    for leader in leaders.values():
        if leader:
            machines.add(leader)
            nodes.add(leader)
            machine_order.append(leader)

    for team in coalitions.values():
        for machine in team:
            machines.add(machine)
            nodes.add(machine)
            machine_order.append(machine)

    for edges in transfers.values():
        for edge in edges:
            edge_id = edge["edge_id"]
            used_edges[edge_id] = edge
            nodes.add(edge["tail"])
            nodes.add(edge["head"])
            outdegree[str(edge["tail"])] += 1
            indegree[str(edge["head"])] += 1
            for host in edge_hosts(edge):
                machines.add(host)
                nodes.add(host)
                machine_order.append(host)

    all_edges = dict(graph_edges)
    all_edges.update({edge_id: edge for edge_id, edge in used_edges.items() if edge_id not in all_edges})
    for edge in all_edges.values():
        nodes.add(edge["tail"])
        nodes.add(edge["head"])
        outdegree[str(edge["tail"])] += 1
        indegree[str(edge["head"])] += 1
        for host in edge_hosts(edge):
            machines.add(host)
            nodes.add(host)
            machine_order.append(host)

    if instance_graph is None:
        entry_order.extend(
            sorted(
                node
                for node in nodes
                if node not in machines and node != "OUT" and outdegree[node] > 0 and indegree[node] == 0
            )
        )

    out_nodes = unique_preserve_order(out_order)
    entry_nodes = unique_preserve_order(entry_order)
    buffers = unique_preserve_order(
        buffer_order
        + sorted(node for node in nodes if node not in machines and node not in out_nodes and node not in entry_nodes)
    )
    return unique_preserve_order(machine_order + sorted(machines)), entry_nodes, buffers, out_nodes, list(all_edges.values())


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


def build_initial_transfers(solution: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    initial_transfers: dict[str, list[dict[str, Any]]] = {}
    for stage_key, edges in solution.get("transfers", {}).items():
        job, op_index = parse_stage_key(stage_key)
        if op_index == 0:
            initial_transfers[job] = order_edges(edges)
    return initial_transfers


def evenly_spaced_positions(count: int) -> list[float]:
    if count <= 1:
        return [0.5]
    return [0.85 - index * (0.70 / (count - 1)) for index in range(count)]


def staggered_machine_positions(count: int, column_index: int) -> list[float]:
    if count <= 0:
        return []

    offsets = [0.08, -0.10, 0.00]
    if count == 1:
        return [max(0.10, min(0.90, 0.5 + offsets[column_index % len(offsets)]))]

    base_positions = evenly_spaced_positions(count)
    return [
        max(0.10, min(0.90, y + offsets[column_index % len(offsets)]))
        for y in base_positions
    ]


def machine_links(
    machines: list[str],
    graph_edges: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[str, set[str]]]:
    machine_set = set(machines)
    links: dict[tuple[str, str], float] = {}
    neighbors: dict[str, set[str]] = {machine: set() for machine in machines}

    for edge in graph_edges:
        tail = str(edge["tail"])
        head = str(edge["head"])
        if tail not in machine_set or head not in machine_set or tail == head:
            continue

        a, b = sorted((tail, head))
        links[(a, b)] = links.get((a, b), 0.0) + 1.0
        neighbors[a].add(b)
        neighbors[b].add(a)

    return links, neighbors


def fallback_machine_layout(machines: list[str]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    machine_columns = [0.20, 0.43, 0.66]
    machine_groups = [machines[index::len(machine_columns)] for index in range(len(machine_columns))]
    for column_index, (column_x, group) in enumerate(zip(machine_columns, machine_groups)):
        for machine, y in zip(group, staggered_machine_positions(len(group), column_index)):
            positions[machine] = (column_x, y)
    return positions


def degree_aware_machine_layout(
    machines: list[str],
    graph_edges: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    links, neighbors = machine_links(machines, graph_edges)
    if not links:
        return fallback_machine_layout(machines)

    degree = {machine: len(neighbors[machine]) for machine in machines}
    max_degree = max(max(degree.values()), 1)
    ordered = sorted(machines, key=lambda item: (-degree[item], item))

    x_min, x_max = 0.14, 0.66
    y_min, y_max = 0.11, 0.89
    center = (0.46, 0.50)
    positions: dict[str, tuple[float, float]] = {}

    for index, machine in enumerate(ordered):
        centrality = degree[machine] / max_degree
        if index == 0:
            positions[machine] = center
            continue

        angle = -math.pi / 2 + 2 * math.pi * (index - 1) / max(len(ordered) - 1, 1)
        radius = 0.20 + 0.18 * (1.0 - centrality)
        x = center[0] + math.cos(angle) * radius * 1.15
        y = center[1] + math.sin(angle) * radius
        positions[machine] = (
            max(x_min, min(x_max, x)),
            max(y_min, min(y_max, y)),
        )

    area = (x_max - x_min) * (y_max - y_min)
    ideal_distance = math.sqrt(area / max(len(machines), 1)) * 0.85
    temperature = 0.055

    for iteration in range(180):
        displacement = {machine: [0.0, 0.0] for machine in machines}

        for left_index, left in enumerate(machines):
            left_x, left_y = positions[left]
            for right in machines[left_index + 1:]:
                right_x, right_y = positions[right]
                dx = left_x - right_x
                dy = left_y - right_y
                distance = max(math.hypot(dx, dy), 1e-6)
                force = (ideal_distance * ideal_distance) / distance
                ux = dx / distance
                uy = dy / distance
                displacement[left][0] += ux * force
                displacement[left][1] += uy * force
                displacement[right][0] -= ux * force
                displacement[right][1] -= uy * force

        for (left, right), weight in links.items():
            left_x, left_y = positions[left]
            right_x, right_y = positions[right]
            dx = right_x - left_x
            dy = right_y - left_y
            distance = max(math.hypot(dx, dy), 1e-6)
            force = (distance * distance / ideal_distance) * min(weight, 2.0)
            ux = dx / distance
            uy = dy / distance
            displacement[left][0] += ux * force
            displacement[left][1] += uy * force
            displacement[right][0] -= ux * force
            displacement[right][1] -= uy * force

        for machine in machines:
            x, y = positions[machine]
            centrality = degree[machine] / max_degree
            center_pull = 0.10 * centrality
            displacement[machine][0] += (center[0] - x) * center_pull
            displacement[machine][1] += (center[1] - y) * center_pull

            radial_x = x - center[0]
            radial_y = y - center[1]
            radial_distance = max(math.hypot(radial_x, radial_y), 1e-6)
            target_radius = 0.13 + 0.27 * (1.0 - centrality)
            shell_force = (target_radius - radial_distance) * 0.05
            displacement[machine][0] += (radial_x / radial_distance) * shell_force
            displacement[machine][1] += (radial_y / radial_distance) * shell_force

        cooling = temperature * (1.0 - iteration / 180)
        for machine in machines:
            dx, dy = displacement[machine]
            distance = max(math.hypot(dx, dy), 1e-6)
            step = min(distance, cooling)
            x, y = positions[machine]
            positions[machine] = (
                max(x_min, min(x_max, x + dx / distance * step)),
                max(y_min, min(y_max, y + dy / distance * step)),
            )

    for machine in machines:
        centrality = degree[machine] / max_degree
        blend = 0.75 * centrality * centrality
        x, y = positions[machine]
        positions[machine] = (
            max(x_min, min(x_max, x * (1.0 - blend) + center[0] * blend)),
            max(y_min, min(y_max, y * (1.0 - blend) + center[1] * blend)),
        )

    return positions


def layout_nodes(
    entry_nodes: list[str],
    machines: list[str],
    buffers: list[str],
    out_nodes: list[str],
    graph_edges: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}

    for entry_name, y in zip(entry_nodes, evenly_spaced_positions(len(entry_nodes))):
        positions[entry_name] = (0.02, y)

    positions.update(degree_aware_machine_layout(machines, graph_edges))

    for buffer_name, y in zip(buffers, evenly_spaced_positions(len(buffers))):
        positions[buffer_name] = (0.88, y)

    for out_name, y in zip(out_nodes, evenly_spaced_positions(len(out_nodes))):
        positions[out_name] = (1.06, y)

    fixed_nodes = entry_nodes + buffers + out_nodes
    for machine in machines:
        machine_x, machine_y = positions[machine]
        for node_name in fixed_nodes:
            node_x, node_y = positions[node_name]
            if abs(machine_y - node_y) > 0.13:
                continue

            min_gap = 0.17 if node_name in buffers else 0.15
            if abs(machine_x - node_x) >= min_gap:
                continue

            if machine_x < node_x:
                machine_x = min(machine_x, node_x - min_gap)
            else:
                machine_x = max(machine_x, node_x + min_gap)
            machine_x = max(0.14, min(0.66, machine_x))

        positions[machine] = (machine_x, machine_y)

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


def compact_edge_label(edge: dict[str, Any]) -> str:
    return f"{edge['tail']}->{edge['head']}"


def label_box(label: str, x: float, y: float) -> tuple[float, float, float, float]:
    width = 0.010 * len(label) + 0.018
    height = 0.035
    return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)


def boxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    padding: float = 0.010,
) -> bool:
    return not (
        left[2] + padding < right[0]
        or right[2] + padding < left[0]
        or left[3] + padding < right[1]
        or right[3] + padding < left[1]
    )


def edge_label_position(
    edge: dict[str, Any],
    start: tuple[float, float],
    end: tuple[float, float],
    lane_index: int,
    lane_count: int,
    occupied_boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(math.hypot(dx, dy), 1e-6)
    perp = (-dy / length, dx / length)
    label = compact_edge_label(edge)

    lane_offset = (lane_index - (lane_count - 1) / 2) * 0.050
    base = lerp(start, end, 0.5)
    candidate_steps = [
        (lane_offset, 0.000),
        (lane_offset + 0.040, 0.018),
        (lane_offset - 0.040, -0.018),
        (lane_offset + 0.080, 0.032),
        (lane_offset - 0.080, -0.032),
        (lane_offset + 0.120, 0.048),
        (lane_offset - 0.120, -0.048),
        (lane_offset + 0.160, 0.064),
        (lane_offset - 0.160, -0.064),
    ]

    best_x = base[0] + perp[0] * lane_offset
    best_y = base[1] + perp[1] * lane_offset + 0.025
    fewest_overlaps = len(occupied_boxes) + 1

    for normal_offset, vertical_offset in candidate_steps:
        x = base[0] + perp[0] * normal_offset
        y = base[1] + perp[1] * normal_offset + 0.025 + vertical_offset
        x = max(-0.03, min(1.10, x))
        y = max(0.05, min(0.95, y))
        box = label_box(label, x, y)
        overlaps = sum(1 for occupied in occupied_boxes if boxes_overlap(box, occupied))
        if overlaps == 0:
            occupied_boxes.append(box)
            return x, y
        if overlaps < fewest_overlaps:
            best_x = x
            best_y = y
            fewest_overlaps = overlaps

    occupied_boxes.append(label_box(label, best_x, best_y))
    return best_x, best_y


def job_state_at(
    stages_by_job: dict[str, list[StageInfo]],
    initial_transfers: dict[str, list[dict[str, Any]]],
    job: str,
    current_time: float,
    positions: dict[str, tuple[float, float]],
    horizon: float,
) -> dict[str, Any]:
    stages = stages_by_job[job]
    first_stage = stages[0]
    initial_edges = initial_transfers.get(job, [])
    if current_time < first_stage.start:
        if initial_edges:
            first_edge = initial_edges[0]
            first_start = float(first_edge["S"])

            if current_time < first_start:
                entry_node = first_edge["tail"]
                return {
                    "position": positions[entry_node],
                    "state": f"{job}:0 waiting on {entry_node}",
                    "group": entry_node,
                    "node": entry_node,
                    "edge_id": None,
                    "active_stage": f"{job}:0",
                    "active_coalition": [],
                    "machine_roles": [],
                }

            for index, edge in enumerate(initial_edges):
                edge_start = float(edge["S"])
                edge_end = edge_start + float(edge["delta"])
                tail_pos = positions[edge["tail"]]
                head_pos = positions[edge["head"]]

                if edge_start <= current_time < edge_end:
                    alpha = (current_time - edge_start) / max(float(edge["delta"]), 1e-9)
                    return {
                        "position": lerp(tail_pos, head_pos, alpha),
                        "state": f"{job}:0 transfer {edge['tail']} -> {edge['head']}",
                        "group": edge["edge_id"],
                        "node": None,
                        "edge_id": edge["edge_id"],
                        "active_stage": f"{job}:0",
                        "active_coalition": [],
                        "machine_roles": [(host, "arm-use") for host in edge_hosts(edge)],
                    }

                next_start = first_stage.start
                if index + 1 < len(initial_edges):
                    next_start = float(initial_edges[index + 1]["S"])

                if edge_end <= current_time < next_start:
                    return {
                        "position": positions[edge["head"]],
                        "state": f"{job}:0 waiting on {edge['head']}",
                        "group": edge["head"],
                        "node": edge["head"],
                        "edge_id": None,
                        "active_stage": f"{job}:0",
                        "active_coalition": [],
                        "machine_roles": [],
                    }

            last_head = initial_edges[-1]["head"]
            return {
                "position": positions[last_head],
                "state": f"{job}:0 waiting for {first_stage.stage_key} on {last_head}",
                "group": last_head,
                "node": last_head,
                "edge_id": None,
                "active_stage": f"{job}:0",
                "active_coalition": [],
                "machine_roles": [],
            }

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
                    "machine_roles": [(host, "arm-use") for host in edge_hosts(edge)],
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
    initial_transfers: dict[str, list[dict[str, Any]]],
    colors: dict[str, Any],
    horizon: float,
) -> tuple[Line2D, list[Patch]]:
    jobs = sorted(stages_by_job)
    row_map = {job: index for index, job in enumerate(jobs)}

    for job in jobs:
        y = row_map[job]
        for edge in initial_transfers.get(job, []):
            timeline_ax.barh(
                y - 0.15,
                float(edge["delta"]),
                left=float(edge["S"]),
                height=0.18,
                color=colors[job],
                edgecolor="black",
                alpha=0.55,
                hatch="//",
            )
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


def render_animation(
    solution: dict[str, Any],
    out_path: Path,
    fps: int,
    time_step: float,
    playback_speed: float,
    instance_graph: dict[str, Any] | None = None,
) -> Path:
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")

    stages_by_job, horizon = build_stage_infos(solution)
    initial_transfers = build_initial_transfers(solution)
    machines, entry_nodes, buffers, out_nodes, used_edges = infer_resources(solution, instance_graph)
    positions = layout_nodes(entry_nodes, machines, buffers, out_nodes, used_edges)
    colors = job_colors(stages_by_job)
    effective_time_step = time_step * playback_speed

    fig = plt.figure(figsize=(14, 8.5))
    network_ax = fig.add_axes([0.05, 0.28, 0.90, 0.67])
    timeline_ax = fig.add_axes([0.08, 0.08, 0.84, 0.14])

    network_ax.set_xlim(-0.08, 1.14)
    network_ax.set_ylim(0.0, 1.0)
    network_ax.set_aspect("equal")
    network_ax.axis("off")
    network_ax.set_title("Animated Solution Flow")

    edge_artists: dict[str, FancyArrowPatch] = {}
    edge_labels: list[Any] = []
    edge_pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    edge_pair_seen: dict[tuple[str, str], int] = defaultdict(int)
    for edge in used_edges:
        pair = tuple(sorted((str(edge["tail"]), str(edge["head"]))))
        edge_pair_counts[pair] += 1

    occupied_label_boxes: list[tuple[float, float, float, float]] = []
    for edge in used_edges:
        start = positions[edge["tail"]]
        end = positions[edge["head"]]
        pair = tuple(sorted((str(edge["tail"]), str(edge["head"]))))
        lane_index = edge_pair_seen[pair]
        edge_pair_seen[pair] += 1
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

        label_pos = edge_label_position(
            edge,
            start,
            end,
            lane_index,
            edge_pair_counts[pair],
            occupied_label_boxes,
        )
        edge_labels.append(
            network_ax.text(
                label_pos[0],
                label_pos[1],
                compact_edge_label(edge),
                ha="center",
                va="center",
                fontsize=6.5,
                color="0.35",
                bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
                zorder=2,
            )
        )

    node_patches: dict[str, Any] = {}
    for entry_name in entry_nodes:
        x, y = positions[entry_name]
        patch = Rectangle(
            (x - 0.06, y - 0.035),
            0.12,
            0.07,
            facecolor="#eef7ff",
            edgecolor="black",
            linewidth=1.8,
            zorder=3,
        )
        network_ax.add_patch(patch)
        node_patches[entry_name] = patch
        network_ax.text(x, y, entry_name, ha="center", va="center", fontsize=9, weight="bold", zorder=4)

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

    time_cursor, timeline_legend = build_timeline_axes(
        timeline_ax, stages_by_job, initial_transfers, colors, horizon
    )
    timeline_ax.legend(handles=timeline_legend, loc="upper right", fontsize=8)

    job_legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[job], markeredgecolor="black", label=job, markersize=9) for job in sorted(colors)]
    network_ax.legend(handles=job_legend, loc="lower left", fontsize=8, title="Jobs")

    frame_count = int(horizon / effective_time_step) + 2

    def update(frame_index: int) -> list[Any]:
        current_time = min(frame_index * effective_time_step, horizon)
        states = {
            job: job_state_at(stages_by_job, initial_transfers, job, current_time, positions, horizon)
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
            elif name in entry_nodes:
                patch.set_facecolor("#eef7ff")
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
            if edge_id not in edge_artists:
                continue
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

        for node_name in entry_nodes + buffers + out_nodes:
            occupants = [job for job, state in states.items() if state["node"] == node_name]
            if occupants and node_name in node_patches:
                lead_job = sorted(occupants)[0]
                node_patches[node_name].set_facecolor(colors[lead_job])
                node_patches[node_name].set_alpha(0.25 if node_name not in out_nodes else 0.35)

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
        "--data",
        default=None,
        help="Path to the original instance JSON. If omitted, outputs/<name>/solution.json tries <name>.json.",
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
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=0.5,
        help="Playback speed multiplier. Values below 1 slow the GIF down.",
    )
    args = parser.parse_args()

    if args.playback_speed <= 0:
        parser.error("--playback-speed must be positive")

    solution_path = Path(args.solution)
    out_path = (
        Path(args.out)
        if args.out is not None
        else solution_path.with_name("solution_animation.gif")
    )

    solution = load_solution(solution_path)
    data_path = Path(args.data) if args.data is not None else default_data_path_for_solution(solution_path)
    instance_graph = load_instance_graph(data_path) if data_path is not None else None
    rendered = render_animation(
        solution,
        out_path,
        fps=args.fps,
        time_step=args.time_step,
        playback_speed=args.playback_speed,
        instance_graph=instance_graph,
    )
    print(f"Wrote: {rendered}")


if __name__ == "__main__":
    main()
