from math import comb

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from fairbench.quantum.dicke import dicke_state


def _check_dicke(n, k):
    probs = Statevector(dicke_state(n, k)).probabilities()
    weights = np.array([bin(i).count("1") for i in range(2**n)])
    assert np.allclose(probs[weights != k], 0.0, atol=1e-10)
    np.testing.assert_allclose(probs[weights == k], 1.0 / comb(n, k), atol=1e-10)


def test_dicke_6_3_exact_uniform():
    _check_dicke(6, 3)
    probs = Statevector(dicke_state(6, 3)).probabilities()
    assert np.sum(probs > 1e-10) == 20


@pytest.mark.parametrize(
    "n,k",
    [(1, 0), (1, 1), (2, 1), (3, 0), (3, 1), (3, 3), (4, 2), (5, 2), (5, 5), (6, 1), (7, 3), (8, 4)],
)
def test_dicke_uniform(n, k):
    _check_dicke(n, k)


def test_dicke_qubit_index_is_asset_index():
    # k=1: amplitude on bitstring with only qubit i set, for every i (Qiskit little-endian index 2**i)
    probs = Statevector(dicke_state(4, 1)).probabilities_dict()
    assert set(probs) == {"0001", "0010", "0100", "1000"}


def test_dicke_no_ancillas_and_no_measure():
    qc = dicke_state(12, 4)
    assert qc.num_qubits == 12 and qc.num_clbits == 0


def test_dicke_rejects_bad_args():
    with pytest.raises(ValueError):
        dicke_state(3, 4)
    with pytest.raises(ValueError):
        dicke_state(3, -1)
