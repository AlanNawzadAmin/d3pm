#!/bin/bash
#SBATCH --job-name=uniref50_process
#SBATCH --output=slurm_out/uniref50_process_%j.out
#SBATCH --error=slurm_out/uniref50_process_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aa11803@nyu.com


# Activate your virtual environment if you're using one
source ~/.bashrc
conda activate evodiff

# Set the working directory
cd /path/to/your/working/directory

tar -xvzf /vast/aa11803/uniref50_data/uniref2020_01.tar.gz -C /vast/aa11803/uniref50_data/
# Run the Python script
python prepare_uniref50.py