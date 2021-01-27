#!/bin/bash
#SBATCH --partition=ib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=1j
#SBATCH --output=output.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=glove1@uvm.edu

source ~/.bashrc
source activate pymoo-CFD
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd ~/Simulations/yales2/pymoo-CFD/YALES
python ./test-runSLURMjobs.py
