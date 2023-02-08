#!/bin/bash
YEARS='2016'
# nodes requested
NODES=4
# time requested
T=2
NUM='all'
LAUNCH='./mpi_launcher.sh'

for yr in `echo $YEARS`
do
	echo $yr
	./mpi_setup.sh $yr $NODES $NUM tmp_"$yr"
	$LAUNCH -P tmp_"$yr" -F -T $T -N "$NODES" -C 28 -p broadwl -J MPI_TMP_"$yr" -M 56
done
rm tmp_"$yr"*
