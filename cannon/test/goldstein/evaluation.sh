#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-01:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=32G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./logs/eval_%j_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/eval_%j_%A_%a.err  # File to which STDERR will be written, %j inserts jobid



# load modules
module load python/3.10.13-fasrc01
mamba activate torch

python evaluation.py