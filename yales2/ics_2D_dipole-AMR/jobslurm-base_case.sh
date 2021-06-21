#!/bin/bash
#SBATCH --partition=ib --constraint="ib&haswell_1|haswell_2|sandybridge"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output-base_case.dat
#SBATCH --job-name=dip-base
#SBATCH --mail-type=END
#SBATCH --mail-user=glove1@uvm.edu

source ~/.bashrc
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd /gpfs1/home/g/l/glove1/pymoo-CFD/yales2/ics_2D_dipole-AMR/base_case/
mpirun 2D_vortex
