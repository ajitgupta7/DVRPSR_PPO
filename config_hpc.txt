#!/usr/bin/zsh

#SBATCH --job-name=DVRPSR20

#SBATCH --output=output.%J.txt

#SBATCH --time=24:00:00
#SBATCH --account=thes1501

#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4

### Ask for four tasks (which are 4 MPI ranks)
#SBATCH --ntasks=4

#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

module load GCCcore/.12.2.0
module load Python/3.10.8
module load cuDNN/8.6.0.163-CUDA-11.8.0

# Insert this AFTER the #SLURM argument section of your job script
export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate base

python run_model.py

