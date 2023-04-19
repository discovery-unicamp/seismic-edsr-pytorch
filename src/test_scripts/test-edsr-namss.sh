# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program execution
# - model
# - data
# - training

#    --save_results \
#    --resume -1 \
#    --test_only \
#    --debug \
#    --reset \

SCALE=2
SEED=2042
TEMPLATE=EDSR_paper

# DIV2K parameters
DATASET=DIV2K_TIFF
DATA_RANGE='1-800/1-50'

# NAMSS parameters
DATASET=NAMSS
DATA_RANGE='1-2746/1-350'

EXPERIMENT_DIR=experiment-01
MODEL_DIR=${TEMPLATE}-X${SCALE}-$DATASET-Seed${SEED}

# Obs: the resume=0 and pre_train arguments allows to test with the best model instead of the latest
python main.py \
    --test_only \
    --save_results \
    --experiment_dir $EXPERIMENT_DIR \
    --load $MODEL_DIR \
    --save $MODEL_DIR \
    --resume 0 \
    --pre_train "../$EXPERIMENT_DIR/$MODEL_DIR/model/model_best.pt" \
    --n_threads 0 \
    --n_GPUs 1 \
    --seed $SEED \
    \
    --template $TEMPLATE \
    --scale $SCALE \
    --shift_mean False \
    \
    --dir_data ../../Data \
    --data_train $DATASET \
    --data_test $DATASET \
    --data_range $DATA_RANGE \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --n_colors 1 \
    \
    --epochs 0 \
