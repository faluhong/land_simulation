#!/bin/sh
#SBATCH --partition=priority  
#SBATCH --account=zhz18039
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array=1-50
#SBATCH --mem=12G
#SBATCH --time=12:00:00  
#SBATCH --exclude=cn498,cn545,cn353,cn366,cn340,cn393,cn544,cn503,cn487,cn333,cn550,cn341,cn361,cn370  

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/LCM_diversity/pythoncode/land_change_modelling/

python3 hispaniola_prediction_overall_parallel.py  --rank=$SLURM_ARRAY_TASK_ID  --n_cores=$SLURM_ARRAY_TASK_MAX 