#!/bin/bash

#SBATCH --cpus-per-task=9
#SBATCH --mem=45G
#SBATCH -t 4-00:00:00
#SBATCH -o %A-%a.out
#SBATCH -e %A-%a.err

python "$@"
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Error"
    exit 100
fi
