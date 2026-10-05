import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from fairbench.backends import bitstrings_to_array, sample


def _xcirc():
    qc = QuantumCircuit(5)
    qc.x(0)
    qc.x(3)
    return qc


@pytest.mark.parametrize("backend", ["aer_statevector", "aer_mps"])
def test_bit_order(backend):
    out = sample(_xcirc(), None, 50, backend=backend, seed=1)
    assert out.shape == (50, 5) and out.dtype == np.uint8
    assert (out == np.array([1, 0, 0, 1, 0])).all()


def test_mps_bond_dim():
    out = sample(_xcirc(), None, 10, backend="aer_mps", seed=1, bond_dim=4, truncation_threshold=1e-10)
    assert (out == np.array([1, 0, 0, 1, 0])).all()


def test_params_list_and_dict():
    th = Parameter("t")
    qc = QuantumCircuit(3)
    qc.ry(th, 1)
    out = sample(qc, [np.pi], 20, seed=0)
    assert (out[:, 1] == 1).all() and (out[:, [0, 2]] == 0).all()
    out = sample(qc, {th: np.pi}, 20, seed=0)
    assert (out[:, 1] == 1).all()


def test_seed_reproducible():
    qc = QuantumCircuit(4)
    qc.h(range(4))
    a = sample(qc, None, 200, seed=7)
    b = sample(qc, None, 200, seed=7)
    assert (a == b).all()


def test_input_not_mutated():
    th = Parameter("t")
    qc = QuantumCircuit(2)
    qc.ry(th, 0)
    ops = qc.count_ops()
    sample(qc, [0.3], 5, seed=0)
    assert qc.count_ops() == ops and qc.num_clbits == 0 and len(qc.parameters) == 1


@pytest.mark.parametrize("b", ["ibm", "vtt"])
def test_hardware_not_implemented(b):
    with pytest.raises(NotImplementedError, match="TOKEN"):
        sample(_xcirc(), None, 1, backend=b)


def test_unknown_backend():
    with pytest.raises(ValueError):
        sample(_xcirc(), None, 1, backend="nope")


def test_bitstrings_to_array():
    out = bitstrings_to_array(["001 10", "00001"], 5)
    assert out.tolist() == [[0, 1, 1, 0, 0], [1, 0, 0, 0, 0]]
