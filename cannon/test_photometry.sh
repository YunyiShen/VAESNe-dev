#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 1-00:30
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ./logs/photometryVAE.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/photometryVAE.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
module load python/3.10.13-fasrc01
source activate torch

python test_photometry.py