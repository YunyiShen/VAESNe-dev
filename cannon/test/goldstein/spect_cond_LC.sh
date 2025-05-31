#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-01:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=64G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./logs/goldstein-mmvae44_%j_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/goldstein-mmvae44_%j_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
#SBATCH --array=0-399

TASK_ID=$SLURM_ARRAY_TASK_ID
TOTAL_TASKS=$SLURM_ARRAY_TASK_COUNT

# load modules
module load python/3.10.13-fasrc01
mamba activate torch

python spect_cond_LC.py --jobid $TASK_ID --totaljobs $TOTAL_TASKS