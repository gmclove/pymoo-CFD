#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output.dat
#SBATCH --job-name=dip-moo
#SBATCH --mail-type=END
#SBATCH --mail-user=glove1@uvm.edu

source ~/.bashrc
source activate pymoo-CFD
module use $HOME/yales2/modules && module load $(cd $HOME/yales2/modules; ls)
cd /gpfs1/home/g/l/glove1/pymoo-CFD/yales2/ics_2D_dipole-AMR
python ./pymooExec.py
