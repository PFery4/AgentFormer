#!/bin/bash

# This directory
DIR="$PWD"

if [[ -z "${PYTHONPATH}" ]]; then
	export PYTHONPATH=$DIR
else
	if [[ ! $PYTHONPATH == *$DIR* ]]; then
		export PYTHONPATH="$PYTHONPATH:$DIR"
	fi
fi

echo "PYTHONPATH variable is:"
echo $PYTHONPATH

# conda environment name
CONDA_ENV="py38-agentformer"

conda activate $CONDA_ENV
