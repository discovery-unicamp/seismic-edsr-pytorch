#!/bin/bash
#
# Obs: needs to be run as BASH!

function get_date () {
    date +%Y-%m-%d_%Hh%Mm%Ss
}

mkdir -p logs
LOG=logs/log-$(get_date).txt

nvidia-smi -L >> $LOG
echo '' >> $LOG

function run_experiment () {
    echo $@ >> $LOG
    echo "Start time: $(get_date)" >> $LOG 
    start=`python -c 'import time ; print(time.time())'`

    bash train_scripts/train-from-params.sh $@ 

    bash test_scripts/test-from-params.sh $@ 

    stop=`python -c 'import time ; print(time.time())'`
    echo "End time: $(get_date)" >> $LOG 
    python -c "print('Duration: {:.2f} min\n'.format(($stop-$start)/60))" >> $LOG
}

DIV2K_PARAMS="DATASET=DIV2K_TIFF TRAIN_DATA_RANGE=1-800/1-20 TEST_DATA_RANGE=1-800/1-50 TEST_EVERY=1000"
NAMSS_PARAMS="DATASET=NAMSS TRAIN_DATA_RANGE=1-2746/1-70 TEST_DATA_RANGE=1-2746/1-350 TEST_EVERY=1030"

EDSR_SEED=
RDN_SEED=
DDBPN_SEED=
WDSR_SEED=
RCAN_SEED=
 
# EDSR DIV2K
run_experiment SCALE=2 SEED=$EDSR_SEED EPOCHS=300 TEMPLATE=EDSR_paper $DIV2K_PARAMS

# EDSR NAMSS
run_experiment SCALE=2 SEED=$EDSR_SEED EPOCHS=300 TEMPLATE=EDSR_paper $NAMSS_PARAMS

# RDN DIV2K
run_experiment SCALE=2 SEED=$RDN_SEED EPOCHS=300 TEMPLATE=RDN $DIV2K_PARAMS

# RDN NAMSS
run_experiment SCALE=2 SEED=$RDN_SEED EPOCHS=300 TEMPLATE=RDN $NAMSS_PARAMS

# DDBPN DIV2K
run_experiment SCALE=2 SEED=$DDBPN_SEED EPOCHS=1000 TEMPLATE=DDBPN $DIV2K_PARAMS

# DDBPN NAMSS
run_experiment SCALE=2 SEED=$DDBPN_SEED EPOCHS=1000 TEMPLATE=DDBPN $NAMSS_PARAMS

# WDSR DIV2K
run_experiment SCALE=2 SEED=$WDSR_SEED EPOCHS=300 TEMPLATE=WDSR $DIV2K_PARAMS

# WDSR NAMSS
run_experiment SCALE=2 SEED=$WDSR_SEED EPOCHS=300 TEMPLATE=WDSR $NAMSS_PARAMS

# RCAN DIV2K
run_experiment SCALE=2 SEED=$RCAN_SEED EPOCHS=300 TEMPLATE=RCAN $DIV2K_PARAMS

# RCAN NAMSS
run_experiment SCALE=2 SEED=$RCAN_SEED EPOCHS=300 TEMPLATE=RCAN $NAMSS_PARAMS
