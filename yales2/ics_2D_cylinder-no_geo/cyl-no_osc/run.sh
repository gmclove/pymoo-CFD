#!/bin/sh

EXEC=2D_cylinder     

if ( [ $# -ne 1 ] ) ; then
  echo "This script requires the number of processors"
  exit
fi

rm -f "$YALES2_HOME"/lib/*.a
make clean

cd "$YALES2_HOME"/src
make -j $1
cd -

make
mpirun -np $1 ./$EXEC

