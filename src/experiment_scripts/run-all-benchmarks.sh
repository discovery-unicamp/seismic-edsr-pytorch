#!/bin/bash
#
# Obs: needs to be run as BASH!

LOG_DIR=logs-benchmark

function get_date () {
    date +%Y-%m-%d_%Hh%Mm%Ss
}

mkdir -p $LOG_DIR
LOG=$LOG_DIR/log-$(get_date).txt

nvidia-smi -L >> $LOG
echo '' >> $LOG

function run_test () {
    echo $@ >> $LOG
    echo "Start time: $(get_date)" >> $LOG 
    start=`python -c 'import time ; print(time.time())'`

    bash experiment_scripts/test-from-params.sh $@ 

    stop=`python -c 'import time ; print(time.time())'`
    echo "End time: $(get_date)" >> $LOG 
    python -c "print('Duration: {:.2f} min\n'.format(($stop-$start)/60))" >> $LOG
}

IFS=$'\n'
declare -a ALL_TESTS=(`python experiment_scripts/get_all_experiments_params.py`)

for TEST_PARAMS in "${ALL_TESTS[@]}"
do
	run_test $TEST_PARAMS 
done ;

