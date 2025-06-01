#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 1-00:30
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o ./logs/photogoldsteinmmvae.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/pphotogoldsteinmmvae.err  # File to which STDERR will be written, %j inserts jobid

module load python/3.10.13-fasrc01
source activate torch

python photometry2goldstein_mmvae.py