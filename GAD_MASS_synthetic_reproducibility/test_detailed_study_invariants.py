from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from synthetic_problem import create_frozen_problem, FrozenSyntheticProblem
from schedules import mass_endpoints_from_tolerances, scalar_cbar_from_terminal_a2


def main():
    p = create_frozen_problem(
        family='broadband_wss', n=24, k=4, graph_seed=11, data_seed=12,
        num_train=64, num_val=16, num_test=16,
    )
    X_snapshot = p.X_train.copy()
    lam1 = p.shifted_eigenvalues(1e-3)
    lam2 = p.shifted_eigenvalues(1e-2)
    assert np.allclose(lam2-lam1, 9e-3)
    np.testing.assert_array_equal(p.X_train, X_snapshot)

    with tempfile.TemporaryDirectory() as td:
        f = Path(td)/'p.npz'; p.save(f); q=FrozenSyntheticProblem.load(f)
        np.testing.assert_array_equal(q.X_train,p.X_train)
        np.testing.assert_array_equal(q.V,p.V)
        assert q.metadata['problem_sha256']==p.metadata['problem_sha256']

    r1 = mass_endpoints_from_tolerances(lam1,p.s_k_train,0.5)
    r2 = mass_endpoints_from_tolerances(lam1,p.s_k_train,1.5)
    assert r1 != r2
    np.testing.assert_array_equal(p.X_train, X_snapshot)

    c1=scalar_cbar_from_terminal_a2(1e-3,1e-3)
    c2=scalar_cbar_from_terminal_a2(1e-2,1e-3)
    assert c1 > c2 > 0
    print('All detailed-study invariants passed.')


if __name__=='__main__':
    main()
