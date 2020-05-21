#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
#ORIGIN=/2nd_disk/workspace-approx/flexfloat-benchmarks/BlackScholes/
ORIGIN=/home/b0rgh/oprecomp/flexfloat-benchmarks/BlackScholes/
cd $ORIGIN
python compile.py $DIR/config_file.txt $DIR/ >/dev/null 2>&1
cd $DIR
./black_scholes2 $1

