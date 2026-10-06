"""Fault-tolerant (FT) resource-estimation helpers: reversible feasibility oracle
(``oracle``) and a classical bit-level simulator for reversible circuits (``revsim``)."""
from .oracle import DEFAULT_BITS, OracleInfo, Quantisation, feasibility_oracle, quantise
from .revsim import simulate, simulate_with_phase

__all__ = ["DEFAULT_BITS", "OracleInfo", "Quantisation", "feasibility_oracle", "quantise",
           "simulate", "simulate_with_phase"]
