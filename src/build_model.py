# src/build_model.py
from __future__ import annotations

from typing import Dict, Tuple, List
import pyomo.environ as pyo

from .data_loader import InstanceData


def compute_bigM(data: InstanceData) -> float:
    """
    Heuristic big-M:
      Sum of worst-case processing times across all operations
      plus worst-case transport per stage.

    This keeps M large enough for disjunctions, but not absurdly huge.
    """
    # max processing time per task type among eligible machines
    max_pt: Dict[str, float] = {}
    for k in data.K:
        max_pt[k] = max(data.ptime[(k, m)] for m in data.M if data.elig[(k, m)] == 1)

    # crude upper bound: each op takes at most max_pt[type]
    proc_bound = 0.0
    for (j, i) in data.ops:
        proc_bound += max_pt[data.tau[(j, i)]]

    # crude upper bound for transport: assume at most |V| edges per stage * max delta
    max_delta = max(data.delta[e] for e in data.E) if data.E else 0.0
    num_stages = sum(len(data.ops_by_job[j]) for j in data.J)  # one stage per op (incl last -> OUT)
    trans_bound = num_stages * (len(data.V) + 1) * max_delta

    return proc_bound + trans_bound + 10.0


def build_model(data: InstanceData) -> pyo.ConcreteModel:
    """
    Builds the full MILP in Pyomo according to your current formulation:

    - Leader selection x_{ijm}
    - Coalition selection y_{ijm} with size r_{tau}
    - Eligibility y <= a
    - Processing time defined by leader
    - Machine capacity across coalition members using disjunctive sequencing
    - Routing as single-path multi-hop flow u_{stage,e} over graph
    - Continuous-time coupling using A/D/S variables
    - Arm capacity: transfers on same arm cannot overlap, duration delta_e
    - Host coupling: if host-machine is busy, its arm also busy, and vice versa
    - Buffer capacity 1: interval non-overlap on [A_{stage,b}, D_{stage,b}]
    """
    m = pyo.ConcreteModel()

    # ----------------------------
    # Sets
    # ----------------------------
    m.J = pyo.Set(initialize=data.J)
    m.MACH = pyo.Set(initialize=data.M)  # avoid name clash with model 'm'
    m.BUF = pyo.Set(initialize=data.B)
    m.K = pyo.Set(initialize=data.K)
    m.V = pyo.Set(initialize=data.V)
    m.E = pyo.Set(initialize=data.E)

    # Operations O = (j,i)
    m.O = pyo.Set(dimen=2, initialize=data.ops)

    # Stages S = also (j,i), one transfer stage after each op i
    # We treat stage (j,i) as "transfer after op i".
    m.S = pyo.Set(dimen=2, initialize=data.ops)

    # Convenience: last operation index per job
    I_last = {j: max(data.ops_by_job[j]) for j in data.J}

    # Helper to get next op for precedence, None if last
    next_op: Dict[Tuple[str, int], Tuple[str, int] | None] = {}
    for (j, i) in data.ops:
        if i < I_last[j]:
            next_op[(j, i)] = (j, i + 1)
        else:
            next_op[(j, i)] = None

    # ----------------------------
    # Parameters
    # ----------------------------
    m.tau = pyo.Param(m.O, initialize=lambda _, j, i: data.tau[(j, i)], within=pyo.Any)

    m.r = pyo.Param(m.K, initialize=lambda _, k: int(data.r[k]), within=pyo.PositiveIntegers)

    m.elig = pyo.Param(m.K, m.MACH, initialize=lambda _, k, mm: int(data.elig[(k, mm)]), within=pyo.Binary)

    m.ptime = pyo.Param(m.K, m.MACH, initialize=lambda _, k, mm: float(data.ptime[(k, mm)]), within=pyo.NonNegativeReals)

    m.release = pyo.Param(m.J, initialize=lambda _, j: float(data.release[j]), within=pyo.NonNegativeReals)
    m.deadline = pyo.Param(m.J, initialize=lambda _, j: float(data.deadline[j]), within=pyo.NonNegativeReals)

    m.tail = pyo.Param(m.E, initialize=lambda _, e: data.tail[e], within=pyo.Any)
    m.head = pyo.Param(m.E, initialize=lambda _, e: data.head[e], within=pyo.Any)
    m.delta = pyo.Param(m.E, initialize=lambda _, e: float(data.delta[e]), within=pyo.NonNegativeReals)

    m.edge_arm = pyo.Param(m.E, initialize=lambda _, e: data.edge_arm[e], within=pyo.Any)
    m.edge_host = pyo.Param(m.E, initialize=lambda _, e: data.edge_host[e], within=pyo.Any)

    # NEU: Parameter für Multi-Capabilities (z.B. A und C gleichzeitig)
    if hasattr(data, "CAPS"):
        m.CAPS = pyo.Set(initialize=data.CAPS)
        m.req_caps = pyo.Param(m.K, m.CAPS, initialize=lambda _, k, c: data.req_caps.get((k, c), 0), within=pyo.NonNegativeIntegers)
        m.has_cap = pyo.Param(m.MACH, m.CAPS, initialize=lambda _, mm, c: data.has_cap.get((mm, c), 0), within=pyo.Binary)

    # NEU: Parameter für Nachbarschaft aus LaTeX (P8)
    if hasattr(data, "neighbor"):
        m.neighbor = pyo.Param(m.MACH, m.MACH, initialize=lambda _, m1, m2: data.neighbor.get((m1, m2), 0), within=pyo.Binary)

    BIGM = compute_bigM(data)
    m.BigM = pyo.Param(initialize=BIGM, within=pyo.PositiveReals)

    # ----------------------------
    # Variables
    # ----------------------------
    # Leader selection
    m.x = pyo.Var(m.O, m.MACH, domain=pyo.Binary)

    # Coalition selection
    m.y = pyo.Var(m.O, m.MACH, domain=pyo.Binary)

    # Times for operations
    m.t = pyo.Var(m.O, domain=pyo.NonNegativeReals)
    m.p = pyo.Var(m.O, domain=pyo.NonNegativeReals)
    m.C = pyo.Var(m.O, domain=pyo.NonNegativeReals)

    # Leader blocking end
    m.g = pyo.Var(m.O, domain=pyo.NonNegativeReals)

    # Resource release time per machine per operation participation
    m.rfree = pyo.Var(m.O, m.MACH, domain=pyo.NonNegativeReals)

    # Routing decisions per stage and edge
    m.u = pyo.Var(m.S, m.E, domain=pyo.Binary)

    # Edge start time
    m.Sedge = pyo.Var(m.S, m.E, domain=pyo.NonNegativeReals)

    # Arrival and departure times at nodes for each stage
    m.Arr = pyo.Var(m.S, m.V, domain=pyo.NonNegativeReals)
    m.Dep = pyo.Var(m.S, m.V, domain=pyo.NonNegativeReals)

    # Sequencing on machines (pairwise, per machine)
    # We'll create q_mach[o1,o2,mm] only for ordered pairs o1<o2 to reduce size.
    op_list = data.ops
    pair_ops = []
    for idx1 in range(len(op_list)):
        for idx2 in range(idx1 + 1, len(op_list)):
            pair_ops.append((op_list[idx1], op_list[idx2]))
    m.PAIR_O = pyo.Set(dimen=4, initialize=[(a[0], a[1], b[0], b[1]) for (a, b) in pair_ops])

    m.q_mach = pyo.Var(m.PAIR_O, m.MACH, domain=pyo.Binary)

    # Arm sequencing for transfers: define alpha=(j,i,e)
    # We'll build pairs only for same arm.
    alpha_list = []
    for (j, i) in data.ops:
        for e in data.E:
            alpha_list.append((j, i, e))
    m.ALPHA = pyo.Set(dimen=3, initialize=alpha_list)

    # q_arm for pairs of alphas that share the same arm, alpha<beta
    alpha_pairs = []
    for a_idx in range(len(alpha_list)):
        for b_idx in range(a_idx + 1, len(alpha_list)):
            aj, ai, ae = alpha_list[a_idx]
            bj, bi, be = alpha_list[b_idx]
            if data.edge_arm[ae] == data.edge_arm[be]:
                alpha_pairs.append((aj, ai, ae, bj, bi, be))
    m.PAIR_A = pyo.Set(dimen=6, initialize=alpha_pairs)
    m.q_arm = pyo.Var(m.PAIR_A, domain=pyo.Binary)

    # Mixed sequencing (host machine vs transfer)
    # This is heavy: q_mix[o, alpha]. For testing it's fine.
    mix_pairs = []
    for (oj, oi) in data.ops:
        for (sj, si) in data.ops:
            for e in data.E:
                mix_pairs.append((oj, oi, sj, si, e))
    m.MIX = pyo.Set(dimen=5, initialize=mix_pairs)
    m.q_mix = pyo.Var(m.MIX, domain=pyo.Binary)

    # Buffer visit indicator
    m.z = pyo.Var(m.S, m.BUF, domain=pyo.Binary)

    # Buffer sequencing pairs
    stage_list = data.ops
    buf_pairs = []
    for b in data.B:
        for a_idx in range(len(stage_list)):
            for c_idx in range(a_idx + 1, len(stage_list)):
                (j1, i1) = stage_list[a_idx]
                (j2, i2) = stage_list[c_idx]
                buf_pairs.append((j1, i1, j2, i2, b))
    m.PAIR_B = pyo.Set(dimen=5, initialize=buf_pairs)
    m.w_buf = pyo.Var(m.PAIR_B, domain=pyo.Binary)

    # ----------------------------
    # Objective: minimize sum of completion times at OUT (Arr at OUT for last stage)
    # ----------------------------
    def obj_rule(mm):
        return sum(mm.Arr[(j, I_last[j]), data.OUT] for j in data.J)

    # m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize) ALTES OBJECTIVE (SUMME ÜBER ALLE)


    # AB HIER: DEFINITION NEUES OBJECTIVE
    m.Tmax = pyo.Var(domain=pyo.NonNegativeReals)

    def makespan_rule(mm, j):
        i = I_last[j]
        return mm.Tmax >= mm.Arr[(j, i), data.OUT]
    m.MAKESPAN = pyo.Constraint(m.J, rule=makespan_rule)

    m.OBJ = pyo.Objective(expr=m.Tmax, sense=pyo.minimize)
    # BIS HIER: DEFINITION NEUES OBJECTIVE

    # ----------------------------
    # (P) Coalition + leader + eligibility
    # ----------------------------
    # P1: exactly one leader
    def one_leader_rule(mm, j, i):
        return sum(mm.x[(j, i), mach] for mach in mm.MACH) == 1
    m.P1 = pyo.Constraint(m.O, rule=one_leader_rule)

    # P2: leader implies coalition membership
    def leader_in_coalition_rule(mm, j, i, mach):
        return mm.x[(j, i), mach] <= mm.y[(j, i), mach]
    m.P2 = pyo.Constraint(m.O, m.MACH, rule=leader_in_coalition_rule)

    # P3: Gesamt-Koalitionsgröße (weiterhin nützlich, um nicht zu viele Maschinen zu binden)
    def coalition_size_rule(mm, j, i):
        k = mm.tau[(j, i)]
        return sum(mm.y[(j, i), mach] for mach in mm.MACH) == mm.r[k]
    m.P3 = pyo.Constraint(m.O, rule=coalition_size_rule)

    # NEU P3b: Multi-Capability-Abdeckung (Ein Task kann z.B. A und C benötigen)
    def capability_mix_rule(mm, j, i, c):
        k = mm.tau[(j, i)]
        req = mm.req_caps[k, c]
        if req == 0:
            return pyo.Constraint.Skip
        # Mindestens 'req' Maschinen in der Koalition müssen die Fähigkeit 'c' haben
        return sum(mm.y[(j, i), mach] * mm.has_cap[mach, c] for mach in mm.MACH) >= req
    
    if hasattr(m, "CAPS"):
        m.P3_caps = pyo.Constraint(m.O, m.CAPS, rule=capability_mix_rule)

    # P4: Generelle Eligibility (Darf die Maschine überhaupt am Task mitarbeiten?)
    def eligibility_rule(mm, j, i, mach):
        k = mm.tau[(j, i)]
        return mm.y[(j, i), mach] <= mm.elig[k, mach]
    m.P4 = pyo.Constraint(m.O, m.MACH, rule=eligibility_rule)

    # NEU P8 aus LaTeX: Räumliche Koalitionsnähe N(m)
    # Maschine m_prime darf nur in die Koalition, wenn sie im Neighborhood N(m) des Leaders m liegt.
    def p8_neighborhood_rule(mm, j, i, m_prime):
        return mm.y[(j, i), m_prime] <= sum(
            mm.x[(j, i), mach]
            for mach in mm.MACH if mm.neighbor[mach, m_prime] == 1
        )
    
    if hasattr(m, "neighbor"):
        m.P8 = pyo.Constraint(m.O, m.MACH, rule=p8_neighborhood_rule)
    # P5: processing time determined by leader
    def proc_time_rule(mm, j, i):
        k = mm.tau[(j, i)]
        return mm.p[(j, i)] == sum(mm.ptime[k, mach] * mm.x[(j, i), mach] for mach in mm.MACH)
    m.P5 = pyo.Constraint(m.O, rule=proc_time_rule)

    # P6: completion time
    def completion_rule(mm, j, i):
        return mm.C[(j, i)] == mm.t[(j, i)] + mm.p[(j, i)]
    m.P6 = pyo.Constraint(m.O, rule=completion_rule)

    # P7: release time for first operation of each job
    def release_rule(mm, j):
        first_i = min(data.ops_by_job[j])
        return mm.t[(j, first_i)] >= mm.release[j]
    m.P7 = pyo.Constraint(m.J, rule=release_rule)

    # ----------------------------
    # (M) Machine blocking/release
    # ----------------------------
    # M1: leader blocking end after completion
    def g_after_C_rule(mm, j, i):
        return mm.g[(j, i)] >= mm.C[(j, i)]
    m.M1 = pyo.Constraint(m.O, rule=g_after_C_rule)

    # M2-M4: define rfree (release time of machine after participating)
    # Helpers: rfree = C
    # Leader:  rfree = g

    def helper_lower_rule(mm, j, i, mach):
        # if y=1 then rfree >= C, else relaxed
        return mm.rfree[(j, i), mach] >= mm.C[(j, i)] - mm.BigM * (1 - mm.y[(j, i), mach])
    m.M2a = pyo.Constraint(m.O, m.MACH, rule=helper_lower_rule)

    def helper_upper_rule(mm, j, i, mach):
        # if y=1 and x=0 -> rfree <= C
        # if x=1 (leader) -> relaxed by +BigM
        return mm.rfree[(j, i), mach] <= mm.C[(j, i)] + mm.BigM * (1 - mm.y[(j, i), mach]) + mm.BigM * mm.x[(j, i), mach]
    m.M2b = pyo.Constraint(m.O, m.MACH, rule=helper_upper_rule)

    def leader_lower_rule(mm, j, i, mach):
        # if x=1 -> rfree >= g
        return mm.rfree[(j, i), mach] >= mm.g[(j, i)] - mm.BigM * (1 - mm.x[(j, i), mach])
    m.M3a = pyo.Constraint(m.O, m.MACH, rule=leader_lower_rule)

    def leader_upper_rule(mm, j, i, mach):
        # if x=1 and y=1 -> rfree <= g (y=1 guaranteed by P2)
        return mm.rfree[(j, i), mach] <= mm.g[(j, i)] + mm.BigM * (1 - mm.x[(j, i), mach]) + mm.BigM * (1 - mm.y[(j, i), mach])
    m.M3b = pyo.Constraint(m.O, m.MACH, rule=leader_upper_rule)

    # M5: machine capacity, disjunctive no-overlap, activated only if both ops use machine
    def mach_no_overlap_1(mm, j1, i1, j2, i2, mach):
        # op1 finishes (rfree) before op2 starts
        q = mm.q_mach[(j1, i1, j2, i2), mach]
        return mm.rfree[(j1, i1), mach] <= mm.t[(j2, i2)] + mm.BigM * (1 - q) + mm.BigM * (2 - mm.y[(j1, i1), mach] - mm.y[(j2, i2), mach])
    def mach_no_overlap_2(mm, j1, i1, j2, i2, mach):
        # op2 finishes before op1 starts
        q = mm.q_mach[(j1, i1, j2, i2), mach]
        return mm.rfree[(j2, i2), mach] <= mm.t[(j1, i1)] + mm.BigM * (q) + mm.BigM * (2 - mm.y[(j1, i1), mach] - mm.y[(j2, i2), mach])

    m.M5a = pyo.Constraint(m.PAIR_O, m.MACH, rule=mach_no_overlap_1)
    m.M5b = pyo.Constraint(m.PAIR_O, m.MACH, rule=mach_no_overlap_2)

    # ----------------------------
    # (F) Routing as flow in graph per stage (j,i)
    # ----------------------------
    # For i<I_last[j]: machines have net flow = x(curr) - x(next)
    # For i==I_last[j]: machines have net flow = x(curr), OUT has -1.
    #
    # Buffers always have net flow 0.

    def flow_machines_rule(mm, j, i, v):
        # v is a machine node
        out_sum = sum(mm.u[(j, i), e] for e in data.out_edges.get(v, []))
        in_sum = sum(mm.u[(j, i), e] for e in data.in_edges.get(v, []))

        nxt = next_op[(j, i)]
        if nxt is not None:
            # net = x(curr,v) - x(next,v)
            return out_sum - in_sum == mm.x[(j, i), v] - mm.x[nxt, v]
        else:
            # last stage: net = x(curr,v)
            return out_sum - in_sum == mm.x[(j, i), v]

    # only for v in M
    m.F1 = pyo.Constraint(m.S, m.MACH, rule=flow_machines_rule)

    def flow_buffers_rule(mm, j, i, b):
        out_sum = sum(mm.u[(j, i), e] for e in data.out_edges.get(b, []))
        in_sum = sum(mm.u[(j, i), e] for e in data.in_edges.get(b, []))
        return out_sum - in_sum == 0
    m.F2 = pyo.Constraint(m.S, m.BUF, rule=flow_buffers_rule)

    # OUT sink for last stage: out - in = -1
    def flow_out_rule(mm, j):
        i = I_last[j]
        out_sum = sum(mm.u[(j, i), e] for e in data.out_edges.get(data.OUT, []))
        in_sum = sum(mm.u[(j, i), e] for e in data.in_edges.get(data.OUT, []))
        return out_sum - in_sum == -1
    m.F3 = pyo.Constraint(m.J, rule=flow_out_rule)

    # forbid using edges into OUT on non-last stages (keeps model consistent)
    def no_out_edges_nonlast_rule(mm, j, i, e):
        if data.head[e] != data.OUT:
            return pyo.Constraint.Skip
        if i == I_last[j]:
            return pyo.Constraint.Skip
        return mm.u[(j, i), e] == 0
    m.F3b = pyo.Constraint(m.S, m.E, rule=no_out_edges_nonlast_rule)

    # F4: no splitting, at most one in/out per node and stage
    def nosplit_out_rule(mm, j, i, v):
        edges = data.out_edges.get(v, [])
        if len(edges) == 0:
            return pyo.Constraint.Feasible  # nichts zu beschränken
        return sum(mm.u[(j, i), e] for e in edges) <= 1

    def nosplit_in_rule(mm, j, i, v):
        edges = data.in_edges.get(v, [])
        if len(edges) == 0:
            return pyo.Constraint.Feasible  # nichts zu beschränken
        return sum(mm.u[(j, i), e] for e in edges) <= 1


    m.F4a = pyo.Constraint(m.S, m.V, rule=nosplit_out_rule)
    m.F4b = pyo.Constraint(m.S, m.V, rule=nosplit_in_rule)

    # ----------------------------
    # (T) Time consistency and process-transport coupling
    # ----------------------------
    # T1: Arr at leader machine equals completion time of op
    def T1_lower(mm, j, i, mach):
        return mm.Arr[(j, i), mach] >= mm.C[(j, i)] - mm.BigM * (1 - mm.x[(j, i), mach])
    def T1_upper(mm, j, i, mach):
        return mm.Arr[(j, i), mach] <= mm.C[(j, i)] + mm.BigM * (1 - mm.x[(j, i), mach])
    m.T1a = pyo.Constraint(m.S, m.MACH, rule=T1_lower)
    m.T1b = pyo.Constraint(m.S, m.MACH, rule=T1_upper)

    # T2: Depart >= Arr at any node (waiting allowed)
    def T2_rule(mm, j, i, v):
        return mm.Dep[(j, i), v] >= mm.Arr[(j, i), v]
    m.T2 = pyo.Constraint(m.S, m.V, rule=T2_rule)

    # T3: If edge e is used, Sedge equals Dep at tail(e)
    def T3_lower(mm, j, i, e):
        tail = data.tail[e]
        return mm.Sedge[(j, i), e] >= mm.Dep[(j, i), tail] - mm.BigM * (1 - mm.u[(j, i), e])
    def T3_upper(mm, j, i, e):
        tail = data.tail[e]
        return mm.Sedge[(j, i), e] <= mm.Dep[(j, i), tail] + mm.BigM * (1 - mm.u[(j, i), e])
    m.T3a = pyo.Constraint(m.S, m.E, rule=T3_lower)
    m.T3b = pyo.Constraint(m.S, m.E, rule=T3_upper)

    # T4: If edge used, Arr at head(e) equals Sedge + delta
    def T4_lower(mm, j, i, e):
        head = data.head[e]
        return mm.Arr[(j, i), head] >= mm.Sedge[(j, i), e] + mm.delta[e] - mm.BigM * (1 - mm.u[(j, i), e])
    def T4_upper(mm, j, i, e):
        head = data.head[e]
        return mm.Arr[(j, i), head] <= mm.Sedge[(j, i), e] + mm.delta[e] + mm.BigM * (1 - mm.u[(j, i), e])
    m.T4a = pyo.Constraint(m.S, m.E, rule=T4_lower)
    m.T4b = pyo.Constraint(m.S, m.E, rule=T4_upper)

    # T5: leader blocking end g equals Dep at leader machine for that stage
    def T5_lower(mm, j, i, mach):
        return mm.g[(j, i)] >= mm.Dep[(j, i), mach] - mm.BigM * (1 - mm.x[(j, i), mach])
    def T5_upper(mm, j, i, mach):
        return mm.g[(j, i)] <= mm.Dep[(j, i), mach] + mm.BigM * (1 - mm.x[(j, i), mach])
    m.T5a = pyo.Constraint(m.S, m.MACH, rule=T5_lower)
    m.T5b = pyo.Constraint(m.S, m.MACH, rule=T5_upper)

    # T6: next operation cannot start before arrival at its leader machine
    def T6_ge_rule(mm, j, i, mach):
        nxt = next_op[(j, i)]
        if nxt is None:
            return pyo.Constraint.Skip
        return mm.t[nxt] >= mm.Arr[(j, i), mach] - mm.BigM * (1 - mm.x[nxt, mach])
    m.T6 = pyo.Constraint(m.S, m.MACH, rule=T6_ge_rule)

    # T6_le: enforce equality when x[nxt, mach] == 1  ->  t[nxt] <= Arr(...)
    def T6_le_rule(mm, j, i, mach):
        nxt = next_op[(j, i)]
        if nxt is None:
            return pyo.Constraint.Skip
        return mm.t[nxt] <= mm.Arr[(j, i), mach] + mm.BigM * (1 - mm.x[nxt, mach])
    m.T6_le = pyo.Constraint(m.S, m.MACH, rule=T6_le_rule)


    # T7: deadline at OUT
    def T7_rule(mm, j):
        i = I_last[j]
        return mm.Arr[(j, i), data.OUT] <= mm.deadline[j]
    m.T7 = pyo.Constraint(m.J, rule=T7_rule)

    # ----------------------------
    # (A) Arm capacity: transfers on same arm cannot overlap
    # ----------------------------
    def A1a_rule(mm, aj, ai, ae, bj, bi, be):
        # alpha before beta
        u_a = mm.u[(aj, ai), ae]
        u_b = mm.u[(bj, bi), be]
        return mm.Sedge[(aj, ai), ae] + mm.delta[ae] <= mm.Sedge[(bj, bi), be] + mm.BigM * (1 - mm.q_arm[(aj, ai, ae, bj, bi, be)]) + mm.BigM * (2 - u_a - u_b)

    def A1b_rule(mm, aj, ai, ae, bj, bi, be):
        # beta before alpha
        u_a = mm.u[(aj, ai), ae]
        u_b = mm.u[(bj, bi), be]
        return mm.Sedge[(bj, bi), be] + mm.delta[be] <= mm.Sedge[(aj, ai), ae] + mm.BigM * (mm.q_arm[(aj, ai, ae, bj, bi, be)]) + mm.BigM * (2 - u_a - u_b)

    m.A1a = pyo.Constraint(m.PAIR_A, rule=A1a_rule)
    m.A1b = pyo.Constraint(m.PAIR_A, rule=A1b_rule)

    # ----------------------------
    # (A2) Host coupling: host machine and its arm are jointly unavailable
    # ----------------------------
    # Disjunction between:
    #   (i) machine occupancy ends before transfer starts: rfree[o,host] <= Sedge[alpha]
    #   (ii) transfer ends before op starts: Sedge[alpha]+delta <= t[o]
    #
    # Activated only if:
    #   machine host participates in operation o (y[o,host]=1) AND transfer alpha is used (u[alpha]=1)
    def A2a_rule(mm, oj, oi, sj, si, e):
        host = data.edge_host[e]
        return mm.rfree[(oj, oi), host] <= mm.Sedge[(sj, si), e] + mm.BigM * (1 - mm.q_mix[(oj, oi, sj, si, e)]) + mm.BigM * (2 - mm.y[(oj, oi), host] - mm.u[(sj, si), e])

    def A2b_rule(mm, oj, oi, sj, si, e):
        host = data.edge_host[e]
        return mm.Sedge[(sj, si), e] + mm.delta[e] <= mm.t[(oj, oi)] + mm.BigM * (mm.q_mix[(oj, oi, sj, si, e)]) + mm.BigM * (2 - mm.y[(oj, oi), host] - mm.u[(sj, si), e])

    m.A2a = pyo.Constraint(m.MIX, rule=A2a_rule)
    m.A2b = pyo.Constraint(m.MIX, rule=A2b_rule)

    # ----------------------------
    # (B) Buffer capacity: capacity 1 via interval non-overlap
    # ----------------------------
    # z[s,b] = sum incoming edges to b (since no-splitting, it's 0/1)
    def B1_rule(mm, j, i, b):
        return mm.z[(j, i), b] == sum(mm.u[(j, i), e] for e in data.in_edges.get(b, []))
    m.B1 = pyo.Constraint(m.S, m.BUF, rule=B1_rule)

    # Non-overlap between stages s1 and s2 on same buffer b
    def B2a_rule(mm, j1, i1, j2, i2, b):
        z1 = mm.z[(j1, i1), b]
        z2 = mm.z[(j2, i2), b]
        w = mm.w_buf[(j1, i1, j2, i2, b)]
        return mm.Arr[(j1, i1), b] >= mm.Dep[(j2, i2), b] - mm.BigM * (1 - w) - mm.BigM * (2 - z1 - z2)

    def B2b_rule(mm, j1, i1, j2, i2, b):
        z1 = mm.z[(j1, i1), b]
        z2 = mm.z[(j2, i2), b]
        w = mm.w_buf[(j1, i1, j2, i2, b)]
        return mm.Arr[(j2, i2), b] >= mm.Dep[(j1, i1), b] - mm.BigM * (w) - mm.BigM * (2 - z1 - z2)

    m.B2a = pyo.Constraint(m.PAIR_B, rule=B2a_rule)
    m.B2b = pyo.Constraint(m.PAIR_B, rule=B2b_rule)

    return m
