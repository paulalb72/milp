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

    # Entry points
    V_in: List[str]
    job_entry: Dict[str, str]

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

    # Multi-capabilities und Nachbarschaft
    CAPS: List[str]
    req_caps: Dict[Tuple[str, str], int]
    has_cap: Dict[Tuple[str, str], int]
    neighbor: Dict[Tuple[str, str], int]

    # Graph
    E: List[str]                                  # edge IDs
    tail: Dict[str, str]
    head: Dict[str, str]
    delta: Dict[str, float]
    edge_arms: Dict[str, List[str]]               # arms serving edge
    arm_host: Dict[str, str]                      # host machine of arm
    edge_hosts: Dict[str, List[str]]              # hosts(edge) = hosts(arms(edge))

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
      - arms define host-machine binding, edge_hosts(e) = hosts(arms(e))
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
    V_in = raw["sets"].get("V_in", [])

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
    job_entry: Dict[str, str] = {}

    for job in raw["jobs"]:
        j = job["job_id"]
        release[j] = float(job.get("release", 0.0))
        deadline[j] = float(job.get("deadline", 1e9))

        # Start-Knoten erfassen (Fallback auf V_in Liste)
        entry = job.get("entry_node")
        if entry:
            job_entry[j] = entry
        elif V_in:
            job_entry[j] = V_in[0]

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
    elig: Dict[Tuple[str, str], int] = {}
    ptime: Dict[Tuple[str, str], float] = {}

    machines = raw["machines"]

    # Multi-capabilities erfassen & Strict Enforcing
    caps_set = set()
    if "CAPS" in raw["sets"]:
        caps_set.update(raw["sets"]["CAPS"])
    for m in M:
        caps_set.update(machines[m].get("capabilities", []))
    CAPS = sorted(list(caps_set))

    has_cap: Dict[Tuple[str, str], int] = {}
    for m in M:
        m_caps = set(machines[m].get("capabilities", []))
        for c in CAPS:
            has_cap[(m, c)] = 1 if c in m_caps else 0

    req_caps: Dict[Tuple[str, str], int] = {}
    for k in K:
        info = raw["task_types"][k]
        
        if "required_capabilities" not in info:
            raise ValueError(f"Strict format enforced: 'required_capabilities' missing for task type '{k}'. Old formats are no longer supported.")
            
        for c in CAPS:
            req_caps[(k, c)] = info["required_capabilities"].get(c, 0)

    for k in K:
        for m in M:
            # Eligible if the machine provides at least one capability required by the task
            can_contribute = any(req_caps[(k, c)] > 0 and has_cap[(m, c)] == 1 for c in CAPS)
            
            if can_contribute:
                elig[(k, m)] = 1
                pt = machines[m].get("processing_time", {})
                if k not in pt:
                    raise ValueError(f"Missing processing_time: machine={m} is eligible for task={k}, but no time is defined for '{k}'")
                ptime[(k, m)] = float(pt[k])
            else:
                elig[(k, m)] = 0
                ptime[(k, m)] = 0.0

    # Nachbarschaftsbeziehungen
    neighbor: Dict[Tuple[str, str], int] = {}
    for m1 in M:
        m1_neighbors = machines[m1].get("neighbors", M)
        for m2 in M:
            neighbor[(m1, m2)] = 1 if m2 in m1_neighbors else 0

    # Arms / host binding
    arm_host: Dict[str, str] = {}
    if "arms" in raw:
        for a, info in raw["arms"].items():
            arm_host[a] = info["host_machine"]

    station_arm: Dict[str, str] = {}
    for a, host in arm_host.items():
        station_arm[host] = a
    for m in M:
        arm_id = machines[m].get("arm_id")
        if machines[m].get("has_arm", False) and arm_id:
            station_arm[m] = arm_id

    # Graph edges
    E: List[str] = []
    tail: Dict[str, str] = {}
    head: Dict[str, str] = {}
    delta: Dict[str, float] = {}
    edge_arms: Dict[str, List[str]] = {}
    edge_hosts: Dict[str, List[str]] = {}

    for e in raw["graph"]["edges"]:
        eid = e["edge_id"]
        E.append(eid)
        tail[eid] = e["tail"]
        head[eid] = e["head"]
        delta[eid] = float(e["delta"])

        arm_val = e.get("served_by_arms", e.get("served_by_arm", []))
        if isinstance(arm_val, str):
            arm_val = [arm_val]
        edge_arms[eid] = arm_val

        tail_arm = station_arm.get(tail[eid])
        head_arm = station_arm.get(head[eid])
        if not tail_arm and not head_arm:
            raise ValueError(
                "Datengrundlage nicht valide, es dürfen keine Kanten existieren, "
                "bei denen weder tail noch head einen Arm haben"
            )
        if tail_arm and head_arm and not {tail_arm, head_arm}.issubset(set(arm_val)):
            raise ValueError(
                "Datengrundlage nicht valide, wenn zwei verbundene Stationen jeweils einen Arm haben, "
                "müssen diese auf der Verbindungsstrecke der Stationen beide genannt werden"
            )

        hosts = []
        for a in arm_val:
            if a not in arm_host:
                raise ValueError(f"Edge {eid} uses arm {a}, but arms[{a}] is not defined in JSON.")
            if arm_host[a] not in hosts:
                hosts.append(arm_host[a])
        edge_hosts[eid] = hosts

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
    # 1) coalition feasibility: check against required capabilities
    for (j, i) in ops:
        k = tau[(j, i)]
        for c in CAPS:
            needed = req_caps[(k, c)]
            if needed > 0:
                capable_m = [m for m in M if has_cap[(m, c)] == 1]
                if len(capable_m) < needed:
                    raise ValueError(
                        f"Infeasible input: operation ({j},{i}) of type {k} needs {needed} "
                        f"machines with capability '{c}', but only {len(capable_m)} have it: {capable_m}"
                    )

    return InstanceData(
        J=J, M=M, B=B, K=K, Arms=Arms, V=V, OUT=OUT,
        V_in=V_in, job_entry=job_entry,
        ops=ops, ops_by_job=ops_by_job, tau=tau,
        release=release, deadline=deadline,
        r=r, elig=elig, ptime=ptime,
        CAPS=CAPS, req_caps=req_caps, has_cap=has_cap, neighbor=neighbor,
        E=E, tail=tail, head=head, delta=delta,
        edge_arms=edge_arms, arm_host=arm_host, edge_hosts=edge_hosts,
        out_edges=out_edges, in_edges=in_edges
    )
