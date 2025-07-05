#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 1-00:30
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -o ./logs/spectraVAE.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/spectraVAE.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=wendysun@mit.edu
module load python/3.10.13-fasrc01
source activate /n/holystore01/LABS/iaifi_lab/Lab/qinyisun/conda/envs/vae_env

python test_spectra.py