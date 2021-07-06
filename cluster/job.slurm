#!/bin/sh
#SBATCH -J lstm
#SBATCH --gres=gpu:1
#SBATCH -p all
#SBATCH -o /mnt/beegfs/home/mergny/time-series-forecasting/cluster/output.out
#SBATCH -e /mnt/beegfs/home/mergny/time-series-forecasting/cluster/error.err

cd ../
echo "Running the job..."
srun python main.py
