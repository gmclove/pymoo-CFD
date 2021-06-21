#!/bin/sh

EXEC=2D_cylinder.py     

if ( [ $# -ne 1 ] ) ; then
  echo "This script requires the number of processors"
  exit
fi

mpirun -np $1 python ./$EXEC

