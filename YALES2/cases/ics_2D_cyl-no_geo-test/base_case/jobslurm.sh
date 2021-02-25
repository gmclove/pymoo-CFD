#!/bin/bash
#SBATCH --partition=ib --constraint="ib&haswell_1|haswell_2|sandybridge"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=base_case
#SBATCH --output=output.dat

source ~/.bashrc
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd ~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cylinder-no_geo/base_case
mpirun 2D_cylinder
