#!/bin/sh
#SBATCH -J lstm
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=all
#SBATCH -t 22:00:00
#SBATCH -o /mnt/beegfs/home/mergny/time-series-forecasting/cluster/output_%j.out
#SBATCH -e /mnt/beegfs/home/mergny/time-series-forecasting/cluster/error_%j.err

cd ../
echo "Running the job..."
srun python main.py

