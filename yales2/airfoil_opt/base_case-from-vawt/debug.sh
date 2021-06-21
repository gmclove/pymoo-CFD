#!/bin/sh

EXEC=2D_af

if ( [ $# -ne 1 ] ) ; then
  echo "This script requires the number of processors"
  exit
fi

echo "WARNING : this script must be used when the option DEBUG_WAIT is equal to 1 in the input file"
read

rm -f "$YALES2_HOME"/lib/*.a
make clean

cd "$YALES2_HOME"/src
make
cd -

make

# clean files
rm -f ps1.txt ps2.txt gdb_cmd.txt
PWD=`pwd`

# gdb commands
touch gdb_cmd.txt
echo "set debug_wait = 0" >> gdb_cmd.txt
echo "continue" >> gdb_cmd.txt

# get the current processes
ps -axwwco pid,user,command >| ps1.txt

# run mpi
mpirun -np $1 ./$EXEC &
sleep 1

# get the new processes
ps -axwwco pid,user,command >| ps2.txt

# find the mpi processes
pids=`diff ps1.txt ps2.txt | awk '{
if ($3 == "'$USER'" && $4 == "'./$EXEC'" )
{
  print $2
}
}'`

# loop on the pids
for apid in $pids
do
  xterm -e gdb -x "$PWD/gdb_cmd.txt" ./$EXEC $apid &
done

# clean files
sleep 5
rm -f ps1.txt ps2.txt gdb_cmd.txt
