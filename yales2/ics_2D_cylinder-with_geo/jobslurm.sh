#!/bin/bash
#SBATCH --partition=ib --constraint="ib&haswell_1|haswell_2|sandybridge"
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.dat
#SBATCH --job-name=cyl-moo
#SBATCH --mail-type=END
#SBATCH --mail-user=glove1@uvm.edu

source ~/.bashrc
source activate pymoo-CFD
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd ~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cylinder-with_geo
python ./pymooExec.py
