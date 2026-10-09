"""Fault-tolerant (FT) resource-estimation helpers.

Qiskit is optional for the analytical/resource-estimation path. The circuit
simulator is imported lazily so data-only and analytic estimators do not need
the circuit backend just to import this package.
"""
from .oracle import DEFAULT_BITS, OracleInfo, Quantisation, feasibility_oracle, quantise


def simulate(circuit, inputs):
    from .revsim import simulate as _simulate
    return _simulate(circuit, inputs)


def simulate_with_phase(circuit, inputs):
    from .revsim import simulate_with_phase as _simulate_with_phase
    return _simulate_with_phase(circuit, inputs)

__all__ = ["DEFAULT_BITS", "OracleInfo", "Quantisation", "feasibility_oracle", "quantise",
           "simulate", "simulate_with_phase"]
