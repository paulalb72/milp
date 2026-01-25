# src/data_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any


@dataclass
class InstanceData:
    # Sets
    J: List[str]
    M: List[str]
    B: List[str]
    K: List[str]
    Arms: List[str]
    V: List[str]
    OUT: str

    # Jobs / operations
    ops: List[Tuple[str, int]]                    # O = [(job_id, op_index)]
    ops_by_job: Dict[str, List[int]]              # {job_id: [1,2,...]}
    tau: Dict[Tuple[str, int], str]               # tau[(j,i)] = task_type

    # Time windows
    release: Dict[str, float]
    deadline: Dict[str, float]

    # Coalition requirement per task type
    r: Dict[str, int]

    # Eligibility and processing time
    elig: Dict[Tuple[str, str], int]              # elig[(k,m)] in {0,1}
    ptime: Dict[Tuple[str, str], float]           # ptime[(k,m)] >=0 (only meaningful if elig=1)

    # Graph
    E: List[str]                                  # edge IDs
    tail: Dict[str, str]
    head: Dict[str, str]
    delta: Dict[str, float]
    edge_arm: Dict[str, str]                      # arm serving edge
    arm_host: Dict[str, str]                      # host machine of arm
    edge_host: Dict[str, str]                     # host(edge) = host(arm(edge))

    # Convenience adjacency
    out_edges: Dict[str, List[str]]               # out_edges[v] = [e ...]
    in_edges: Dict[str, List[str]]                # in_edges[v]  = [e ...]


def load_instance(json_path: str | Path) -> InstanceData:
    """
    Reads your JSON (test.json) and converts it into a structured instance object.

    Semantics:
      - jobs define operations and their task types tau_{ij}
      - task_types define coalition_size r_k
      - machines define capabilities -> elig matrix a_{k,m}
      - graph edges define allowed transfers with durations delta_e
      - arms define host-machine binding, edge_host(e) = host(arm(e))
    """
    path = Path(json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Sets
    J = raw["sets"]["J"]
    M = raw["sets"]["M"]
    B = raw["sets"]["B"]
    K = raw["sets"]["K"]
    Arms = raw["sets"].get("A", [])
    OUT = raw["sets"].get("V_out", "OUT")

    # Nodes: we accept either explicit nodes or infer
    graph_nodes = raw["graph"].get("nodes", None)
    if graph_nodes is None:
        V = list(M) + list(B) + [OUT]
    else:
        V = graph_nodes

    # Jobs and operations
    ops: List[Tuple[str, int]] = []
    ops_by_job: Dict[str, List[int]] = {}
    tau: Dict[Tuple[str, int], str] = {}
    release: Dict[str, float] = {}
    deadline: Dict[str, float] = {}

    for job in raw["jobs"]:
        j = job["job_id"]
        release[j] = float(job.get("release", 0.0))
        deadline[j] = float(job.get("deadline", 1e9))

        op_list = job["operations"]
        op_indices = [int(o["op_index"]) for o in op_list]
        ops_by_job[j] = sorted(op_indices)

        for o in op_list:
            i = int(o["op_index"])
            k = o["task_type"]
            ops.append((j, i))
            tau[(j, i)] = k

    ops = sorted(ops, key=lambda x: (x[0], x[1]))

    # Task types -> coalition sizes r_k
    r: Dict[str, int] = {}
    for k, info in raw["task_types"].items():
        r[k] = int(info["coalition_size"])

    # Machines -> capabilities and processing times
    # Build elig[(k,m)] and ptime[(k,m)]
    elig: Dict[Tuple[str, str], int] = {}
    ptime: Dict[Tuple[str, str], float] = {}

    machines = raw["machines"]
    for k in K:
        for m in M:
            elig[(k, m)] = 0
            ptime[(k, m)] = 0.0

    for m in M:
        caps = set(machines[m].get("capabilities", []))
        pt = machines[m].get("processing_time", {})

        for k in K:
            if k in caps:
                elig[(k, m)] = 1
                if k not in pt:
                    raise ValueError(f"Missing processing_time for capability: machine={m}, task={k}")
                ptime[(k, m)] = float(pt[k])
            else:
                elig[(k, m)] = 0
                ptime[(k, m)] = 0.0  # unused if elig=0

    # Arms / host binding
    arm_host: Dict[str, str] = {}
    if "arms" in raw:
        for a, info in raw["arms"].items():
            arm_host[a] = info["host_machine"]

    # Graph edges
    E: List[str] = []
    tail: Dict[str, str] = {}
    head: Dict[str, str] = {}
    delta: Dict[str, float] = {}
    edge_arm: Dict[str, str] = {}
    edge_host: Dict[str, str] = {}

    for e in raw["graph"]["edges"]:
        eid = e["edge_id"]
        E.append(eid)
        tail[eid] = e["tail"]
        head[eid] = e["head"]
        delta[eid] = float(e["delta"])
        edge_arm[eid] = e["served_by_arm"]

        a = edge_arm[eid]
        if a not in arm_host:
            raise ValueError(f"Edge {eid} uses arm {a}, but arms[{a}] is not defined in JSON.")
        edge_host[eid] = arm_host[a]

    # Build adjacency lists
    out_edges: Dict[str, List[str]] = {v: [] for v in V}
    in_edges: Dict[str, List[str]] = {v: [] for v in V}
    for eid in E:
        u = tail[eid]
        v = head[eid]
        if u not in out_edges:
            out_edges[u] = []
        if v not in in_edges:
            in_edges[v] = []
        out_edges[u].append(eid)
        in_edges[v].append(eid)

    # Basic feasibility checks (very useful for debugging)
    # 1) coalition feasibility: for each op, enough eligible machines exist
    for (j, i) in ops:
        k = tau[(j, i)]
        need = r[k]
        eligible_m = [m for m in M if elig[(k, m)] == 1]
        if len(eligible_m) < need:
            raise ValueError(
                f"Infeasible input: operation ({j},{i}) has type {k} requiring r={need}, "
                f"but only {len(eligible_m)} machines are eligible: {eligible_m}"
            )

    return InstanceData(
        J=J, M=M, B=B, K=K, Arms=Arms, V=V, OUT=OUT,
        ops=ops, ops_by_job=ops_by_job, tau=tau,
        release=release, deadline=deadline,
        r=r, elig=elig, ptime=ptime,
        E=E, tail=tail, head=head, delta=delta,
        edge_arm=edge_arm, arm_host=arm_host, edge_host=edge_host,
        out_edges=out_edges, in_edges=in_edges
    )
