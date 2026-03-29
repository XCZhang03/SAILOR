#!/bin/bash
#SBATCH --job-name run_mpc
#SBATCH -c 4               # Number of cores (-c)
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_test # Partition to submit to
#SBATCH --gres=gpu:1        # Number of GPUs (per node)
#SBATCH --mem=100g   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o scratch_dir/slurm_logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e scratch_dir/slurm_logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -A kempner_ydu_lab

module load python

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR

mamba activate ei_world_model
cd large-video-planner
python server.py & 
sleep 120
cd ..

mamba activate libero_env
python dp_search.py