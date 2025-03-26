#!/bin/sh
#SBATCH --account=zhz18039
#SBATCH --partition=general
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --time=12:00:00                      

source /home/fah20002/miniconda3/etc/profile.d/conda.sh
conda activate py38

cd /scratch/zhz18039/fah20002/LCM_diversity/pythoncode/land_change_modelling/

python3 change_matrix_hispaniola.py   --outputpath_version_flag='degrade_v2_refine_3_3'  --folder='mosaic'   --predict_flag='hindcast'   --country_flag='hispaniola'