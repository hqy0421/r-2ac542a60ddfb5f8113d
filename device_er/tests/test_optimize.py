"""Global optimisation: exactness, consistency, constraints, and the
explicit difference from threshold connected-components."""

import itertools

import numpy as np
import pytest

from device_er.errors import OptimizationError
from device_er.optimize import (
    _solve_enumeration,
    _solve_milp,
    _pairs_of,
    solve_partition,
)

C_FM, C_DM = 6.0, 1.5


def _probs(ids, table):
    return {(min(a, b), max(a, b)): p for (a, b), p in table.items()}


def test_chain_differs_from_threshold_connected_components():
    # p12 = p23 = 0.9 > tau = 0.8, p13 = 0.1: CC merges all three,
    # the global optimum splits the chain
    ids = [1, 2, 3]
    probs = _probs(ids, {(1, 2): 0.9, (2, 3): 0.9, (1, 3): 0.1})
    result = solve_partition(ids, probs, C_FM, C_DM)
    assert result.clusters != [[1, 2, 3]]
    expected = 0.6 + 0.9 * C_DM + 0.1 * C_DM  # one merge + two splits
    assert result.total_risk == pytest.approx(expected, abs=1e-9)
    cc = result.baselines["threshold_connected_components"]
    assert cc["total_risk"] == pytest.approx(0.6 + 0.6 + 0.9 * C_FM, abs=1e-9)
    assert result.total_risk < cc["total_risk"]


def test_partition_is_transitive_for_random_instances():
    rng = np.random.default_rng(3)
    for _ in range(6):
        ids = list(range(1, 9))
        probs = {(a, b): float(rng.random())
                 for a, b in itertools.combinations(ids, 2)}
        result = solve_partition(ids, probs, C_FM, C_DM)
        entity = result.archive_to_entity
        for a, b, c in itertools.permutations(ids, 3):
            if entity[a] == entity[b] and entity[b] == entity[c]:
                assert entity[a] == entity[c]


def test_must_link_and_cannot_link_are_hard():
    ids = [1, 2, 3]
    probs = _probs(ids, {(1, 2): 0.99, (2, 3): 0.99, (1, 3): 0.99})
    result = solve_partition(ids, probs, C_FM, C_DM, cannot_link=[(1, 3)])
    entity = result.archive_to_entity
    assert entity[1] != entity[3]
    assert entity[1] == entity[2] or entity[2] == entity[3]

    result = solve_partition(ids, probs, C_FM, C_DM, must_link=[(1, 3)],
                             cannot_link=[(1, 2)])
    entity = result.archive_to_entity
    assert entity[1] == entity[3]
    assert entity[1] != entity[2]


def test_infeasible_constraints_raise():
    ids = [1, 2, 3]
    probs = _probs(ids, {(1, 2): 0.5, (2, 3): 0.5, (1, 3): 0.5})
    with pytest.raises(OptimizationError):
        solve_partition(ids, probs, C_FM, C_DM,
                        must_link=[(1, 2), (2, 3)], cannot_link=[(1, 3)])
    with pytest.raises(OptimizationError):
        solve_partition(ids, probs, C_FM, C_DM,
                        must_link=[(1, 2)], cannot_link=[(2, 1)])


def test_milp_matches_exact_enumeration():
    rng = np.random.default_rng(11)
    n = 7
    pairs = _pairs_of(n)
    coef = rng.normal(size=len(pairs))
    constant = float(rng.random())
    x_milp, _ = _solve_milp(n, pairs, coef, [], [])
    x_enum, _ = _solve_enumeration(n, pairs, coef, constant, [], [])
    assert constant + coef @ x_milp == pytest.approx(
        constant + coef @ x_enum, abs=1e-6
    )


def test_global_optimum_beats_all_baselines():
    rng = np.random.default_rng(5)
    ids = list(range(1, 11))
    probs = {(a, b): float(rng.random())
             for a, b in itertools.combinations(ids, 2)}
    result = solve_partition(ids, probs, C_FM, C_DM)
    assert result.total_risk <= result.baselines["all_singletons"]["total_risk"] + 1e-9
    assert result.total_risk <= result.baselines["all_merged"]["total_risk"] + 1e-9
    cc = result.baselines["threshold_connected_components"]
    assert result.total_risk <= cc["total_risk"] + 1e-9


def test_invalid_inputs_raise():
    with pytest.raises(OptimizationError):
        solve_partition([], {}, C_FM, C_DM)
    with pytest.raises(OptimizationError):
        solve_partition([1, 2], {(1, 2): 1.5}, C_FM, C_DM)
    with pytest.raises(OptimizationError):
        solve_partition([1, 2], {(1, 2): 0.5}, -1.0, C_DM)
