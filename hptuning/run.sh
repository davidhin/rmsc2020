#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=15:00:00
#SBATCH --mem=16000MB
#SBATCH --array=1-312
#SBATCH --err="outputs/MyJob_%a.err"
#SBATCH --output="outputs/MyJob_%a.out"
#SBATCH --job-name="tp-dh"

## Setup Python Environment
module load Anaconda3/5.0.1
module load Java/10.0.1
source activate main_gpu

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p input.csv`
python3 main.py "${par[0]}" "${par[1]}" "${par[2]}"
