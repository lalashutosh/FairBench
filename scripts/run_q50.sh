#!/bin/bash
# FairBench quantum core on VTT Q50 via LUMI (Quantum x Finance hackathon values,
# from QxF-LUMI-AI-Factory-QuickStart-Q50.pdf). Run from the repo root on LUMI:
#   sbatch scripts/run_q50.sh            # raw hardware
#   EMS=1 sbatch scripts/run_q50.sh      # with FiQCI-EMS mitigation level 1
# Output: results/hw/hw_q50_q50_*.json (+ q50-<jobid>.out). Quantum jobs are limited to 2 h.
#SBATCH --job-name=fairbench-q50
#SBATCH --account=project_462001763
#SBATCH --reservation=quantumfinance
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=q50-%j.out

module use /appl/local/quantum/modulefiles/hackathon
module --ignore_cache load fiqci-vtt-qiskit-QxF

python scripts/hw_q50.py --backend q50 --exp all --shots ${SHOTS:-2000} ${EMS:+--ems $EMS}
