"""Globally minimum-risk consistent partitioning.

Given pairwise match probabilities ``p_ij``, a false-merge cost
``C_FM`` and a missed-merge cost ``C_DM``, the expected pairwise risk
of a clustering encoded by ``x_ij in {0,1}`` (1 = same entity) is

    R(x) = sum_ij [ x_ij * (1 - p_ij) * C_FM + (1 - x_ij) * p_ij * C_DM ]

We minimise R(x) exactly over *consistent* clusterings - x must satisfy
the triangle inequalities (transitivity), so the result is always a
valid partition, never a contradictory threshold graph.  Must-link and
cannot-link relations are hard constraints.  Solved as a MILP via
scipy/HiGHS; for tiny inputs an exact set-partition enumeration is the
fallback.  This is deliberately NOT threshold + connected components;
that heuristic is only computed as a baseline in the risk report.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import OptimizationError


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class PartitionResult:
    archive_ids: list
    clusters: list  # list of sorted member-id lists
    assignment: dict  # (id_a, id_b) -> 0/1
    total_risk: float
    merge_risk: float
    split_risk: float
    objective_constant: float
    solver: str
    pair_table: pd.DataFrame
    baselines: dict = field(default_factory=dict)
    must_link: list = field(default_factory=list)
    cannot_link: list = field(default_factory=list)

    @property
    def archive_to_entity(self):
        return {a: min(members) for members in self.clusters for a in members}


def _pairs_of(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _check_constraints(index, must_link, cannot_link, ids):
    must = {(min(a, b), max(a, b)) for a, b in must_link}
    cannot = {(min(a, b), max(a, b)) for a, b in cannot_link}
    overlap = must & cannot
    if overlap:
        raise OptimizationError(
            "same pair declared both must_link and cannot_link",
            detail={"pairs": sorted(overlap)},
        )
    uf = _UnionFind(len(ids))
    for a, b in must:
        uf.union(index[a], index[b])
    for a, b in cannot:
        if uf.find(index[a]) == uf.find(index[b]):
            raise OptimizationError(
                "infeasible constraints: cannot_link pair is forced together "
                "by the must-link closure",
                detail={"cannot_link_pair": [a, b]},
            )
    return sorted(must), sorted(cannot)


def _solve_milp(n, pairs, coef, must_idx, cannot_idx):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    m = len(pairs)
    pair_pos = {p: k for k, p in enumerate(pairs)}
    triples = list(itertools.combinations(range(n), 3))
    A = lil_matrix((3 * len(triples), m), dtype=float)
    row = 0
    for i, j, k in triples:
        ij, ik, jk = pair_pos[(i, j)], pair_pos[(i, k)], pair_pos[(j, k)]
        # x_ij + x_jk - x_ik <= 1   (and the two symmetric forms)
        A[row, ij] = 1.0; A[row, jk] = 1.0; A[row, ik] = -1.0; row += 1
        A[row, ij] = 1.0; A[row, ik] = 1.0; A[row, jk] = -1.0; row += 1
        A[row, ik] = 1.0; A[row, jk] = 1.0; A[row, ij] = -1.0; row += 1
    constraints = LinearConstraint(
        A.tocsr(), lb=np.full(row, -np.inf), ub=np.ones(row)
    )
    lb, ub = np.zeros(m), np.ones(m)
    for k in must_idx:
        lb[k] = ub[k] = 1.0
    for k in cannot_idx:
        lb[k] = ub[k] = 0.0
    result = milp(
        c=coef,
        constraints=constraints,
        bounds=Bounds(lb, ub),
        integrality=np.ones(m),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise OptimizationError(
            f"MILP solver failed: {getattr(result, 'message', 'unknown error')}"
        )
    return (np.asarray(result.x) > 0.5).astype(int), "scipy-milp(highs)"


def _solve_enumeration(n, pairs, coef, constant, must_idx, cannot_idx):
    """Exact minimum over all set partitions (restricted growth strings)."""
    must, cannot = set(must_idx), set(cannot_idx)
    pair_pos = {p: k for k, p in enumerate(pairs)}
    best_x, best_obj = None, np.inf
    assignment = [0] * n

    def objective():
        x = np.array(
            [1 if assignment[i] == assignment[j] else 0 for i, j in pairs]
        )
        return constant + float(coef @ x), x

    def feasible():
        for k in must:
            i, j = pairs[k]
            if assignment[i] != assignment[j]:
                return False
        for k in cannot:
            i, j = pairs[k]
            if assignment[i] == assignment[j]:
                return False
        return True

    def rec(pos, n_blocks):
        nonlocal best_x, best_obj
        if pos == n:
            if feasible():
                obj, x = objective()
                if obj < best_obj - 1e-12:
                    best_obj, best_x = obj, x
            return
        for block in range(n_blocks + 1):
            assignment[pos] = block
            rec(pos + 1, max(n_blocks, block + 1))

    rec(1, 1)
    if best_x is None:
        raise OptimizationError("no feasible partition satisfies the constraints")
    return best_x, "exact-enumeration"


def _clusters_from_x(ids, pairs, x):
    uf = _UnionFind(len(ids))
    for k, (i, j) in enumerate(pairs):
        if x[k]:
            uf.union(i, j)
    groups = {}
    for pos, archive_id in enumerate(ids):
        groups.setdefault(uf.find(pos), []).append(archive_id)
    return sorted((sorted(m) for m in groups.values()), key=lambda m: m[0])


def _risk_split(cost_merge, cost_split, x):
    merge = float(cost_merge[x == 1].sum())
    split = float(cost_split[x == 0].sum())
    return merge, split, merge + split


def solve_partition(
    archive_ids,
    pair_probs: dict,
    false_merge_cost: float,
    missed_merge_cost: float,
    must_link=(),
    cannot_link=(),
) -> PartitionResult:
    """Exact minimum-expected-risk consistent partition.

    ``pair_probs`` maps ``(id_a, id_b)`` (a < b) -> P(same entity).
    """
    ids = sorted(int(a) for a in archive_ids)
    n = len(ids)
    if n == 0:
        raise OptimizationError("empty archive set")
    index = {a: i for i, a in enumerate(ids)}
    must, cannot = _check_constraints(index, must_link, cannot_link, ids)

    pairs = _pairs_of(n)
    pair_index = {(ids[i], ids[j]): k for k, (i, j) in enumerate(pairs)}
    try:
        p = np.array([float(pair_probs[(ids[i], ids[j])]) for i, j in pairs])
    except KeyError as exc:
        raise OptimizationError(f"missing pairwise probability for {exc}") from exc
    if np.any((p < 0) | (p > 1)):
        raise OptimizationError("pairwise probabilities outside [0, 1]")

    c_fm, c_dm = float(false_merge_cost), float(missed_merge_cost)
    if c_fm <= 0 or c_dm <= 0:
        raise OptimizationError("merge costs must be strictly positive")
    cost_merge = (1.0 - p) * c_fm  # risk incurred when the pair is merged
    cost_split = p * c_dm          # risk incurred when the pair is split
    coef = cost_merge - cost_split
    constant = float(cost_split.sum())

    must_idx = [pair_index[(ids[i], ids[j])] for (i, j) in
                [(index[a], index[b]) for a, b in must]]
    cannot_idx = [pair_index[(ids[i], ids[j])] for (i, j) in
                  [(index[a], index[b]) for a, b in cannot]]

    if n == 1:
        x = np.zeros(0, dtype=int)
        solver = "trivial"
    else:
        try:
            x, solver = _solve_milp(n, pairs, coef, must_idx, cannot_idx)
        except ImportError:  # pragma: no cover - scipy always present here
            if n > 9:
                raise OptimizationError(
                    f"no MILP solver available and n={n} too large for enumeration"
                )
            x, solver = _solve_enumeration(n, pairs, coef, constant,
                                           must_idx, cannot_idx)

    # the rounded solution must be an exactly consistent partition
    pair_pos = {pr: k for k, pr in enumerate(pairs)}
    for i, j, k in itertools.combinations(range(n), 3):
        ij, ik, jk = (x[pair_pos[(i, j)]], x[pair_pos[(i, k)]], x[pair_pos[(j, k)]])
        if ij + jk - ik > 1 or ij + ik - jk > 1 or ik + jk - ij > 1:
            raise OptimizationError(
                "solver produced a non-transitive clustering - refusing to apply"
            )
    for k in must_idx:
        if not x[k]:
            raise OptimizationError("solver violated a must_link constraint")
    for k in cannot_idx:
        if x[k]:
            raise OptimizationError("solver violated a cannot_link constraint")

    clusters = _clusters_from_x(ids, pairs, x)
    merge_risk, split_risk, total_risk = _risk_split(cost_merge, cost_split, x)

    # ---------------- baselines for the risk explanation ----------------
    baselines = {}
    m0, s0, r0 = _risk_split(cost_merge, cost_split, np.zeros_like(x))
    baselines["all_singletons"] = {"total_risk": r0, "merge_risk": m0,
                                   "split_risk": s0, "feasible": True}
    if cannot:
        baselines["all_merged"] = {"feasible": False,
                                   "reason": "violates cannot_link constraints"}
    else:
        m1, s1, r1 = _risk_split(cost_merge, cost_split, np.ones_like(x))
        baselines["all_merged"] = {"total_risk": r1, "merge_risk": m1,
                                   "split_risk": s1, "feasible": True}
    tau = c_fm / (c_fm + c_dm)
    cc_uf = _UnionFind(n)
    for k_, (i, j) in enumerate(pairs):
        if p[k_] >= tau:
            cc_uf.union(i, j)
    x_cc = np.array(
        [1 if cc_uf.find(i) == cc_uf.find(j) else 0 for i, j in pairs]
    )
    mcc, scc, rcc = _risk_split(cost_merge, cost_split, x_cc)
    baselines["threshold_connected_components"] = {
        "threshold": tau,
        "total_risk": rcc,
        "merge_risk": mcc,
        "split_risk": scc,
        "must_link_violations": int(sum(1 for k_ in must_idx if not x_cc[k_])),
        "cannot_link_violations": int(sum(1 for k_ in cannot_idx if x_cc[k_])),
        "feasible": bool(
            all(x_cc[k_] for k_ in must_idx)
            and not any(x_cc[k_] for k_ in cannot_idx)
        ),
    }

    constraint_kind = {}
    for a, b in must:
        constraint_kind[(a, b)] = "must_link"
    for a, b in cannot:
        constraint_kind[(a, b)] = "cannot_link"
    pair_table = pd.DataFrame(
        {
            "archive_id_a": [ids[i] for i, _ in pairs],
            "archive_id_b": [ids[j] for _, j in pairs],
            "prob": p,
            "decision": ["merge" if v else "split" for v in x],
            "risk": np.where(x == 1, cost_merge, cost_split),
            "constraint": [
                constraint_kind.get((ids[i], ids[j]), "") for i, j in pairs
            ],
        }
    )

    return PartitionResult(
        archive_ids=ids,
        clusters=clusters,
        assignment={(ids[i], ids[j]): int(x[k]) for k, (i, j) in enumerate(pairs)},
        total_risk=total_risk,
        merge_risk=merge_risk,
        split_risk=split_risk,
        objective_constant=constant,
        solver=solver,
        pair_table=pair_table,
        baselines=baselines,
        must_link=[list(t) for t in must],
        cannot_link=[list(t) for t in cannot],
    )
