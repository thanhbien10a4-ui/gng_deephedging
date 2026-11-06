#!/bin/bash
#SBATCH --job-name=dh_smoke24
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/user04/gng_deephedging/deephedging-rewritten/tests/slurm_smoke_%j.out
#SBATCH --error=/home/user04/gng_deephedging/deephedging-rewritten/tests/slurm_smoke_%j.err

echo "=== Slurm Job Starting ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

cd /home/user04/gng_deephedging/deephedging-rewritten
/home/user04/gng_deephedging/.venv/bin/python tests/test_smoke_24.py

EXIT_CODE=$?
echo "=== Slurm Job Finished with exit code: $EXIT_CODE ==="
exit $EXIT_CODE
