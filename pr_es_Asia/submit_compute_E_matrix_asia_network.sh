#!/bin/bash

#SBATCH --partition=cpu-short
#SBATCH --array=0-7
#SBATCH --job-name=event_sync_mlcs
#SBATCH --output=../jobs/event_sync_%j.out
#SBATCH --error=../jobs/event_sync_%j.err
##SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --mail-user=felix.strnad@uni-tuebingen.de
#SBATCH --mail-type=ALL

export PYTHONPATH=/home/strnad/multiclimnet/
python create_asia_network.py

