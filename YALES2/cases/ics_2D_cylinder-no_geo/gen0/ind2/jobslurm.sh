#!/bin/bash
#SBATCH --partition=ib --constraint="ib&haswell_2"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=g0.i2
#SBATCH --output=output.dat

source ~/.bashrc
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd ~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cylinder-no_geo/gen0/ind2
mpirun 2D_cylinder
