#!/bin/sh
#SBATCH --partition=priority
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-30
#SBATCH --mem=24G
#SBATCH --time=12:00:00

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/LCM_diversity/pythoncode/change_probability_map/    # replace with your own path

python3 prepare_variable_multi_lc.py  --rank=$SLURM_ARRAY_TASK_ID  --n_cores=$SLURM_ARRAY_TASK_MAX



