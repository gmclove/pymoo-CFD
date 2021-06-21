#!/bin/bash
#SBATCH --partition=ib --constraint="ib&haswell_1|haswell_2|sandybridge"
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=cyl-base
#SBATCH --output=output.dat

source ~/.bashrc
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd ~/pymoo-CFD/yales2/ics_2D_cylinder-no_geo/base_case/
mpirun ./2D_cylinder
