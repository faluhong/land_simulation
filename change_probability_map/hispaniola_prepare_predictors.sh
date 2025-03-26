#!/bin/sh
#SBATCH --account=zhz18039
#SBATCH --partition=priority
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00                

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/LCM_diversity/pythoncode/land_change_modelling/

python3 hispaniola_prepare_predictors.py