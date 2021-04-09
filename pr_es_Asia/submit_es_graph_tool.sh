#!/bin/bash

#SBATCH --partition=cpu-short
#SBATCH --array=0-30
#SBATCH --job-name=es_graph_tool_mlcs
#SBATCH --output=../jobs/es_graph_tool_%j.out
#SBATCH --error=../jobs/es_graph_tool_%j.err
##SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --mail-user=felix.strnad@uni-tuebingen.de
#SBATCH --mail-type=ALL

export PYTHONPATH=/home/strnad/multiclimnet
python clustering_asia.py

