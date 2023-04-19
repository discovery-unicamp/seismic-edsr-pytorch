
# Run setting the following parameters appropriately:
#
# bash train-from-parameters.sh EXPERIMENT_NUM= SCALE= SEED= EPOCHS= TEMPLATE= DATASET= TRAIN_DATA_RANGE= TEST_EVERY=

declare $@

EXPERIMENT_DIR=experiment-$EXPERIMENT_NUM
MODEL_DIR=${TEMPLATE}-X${SCALE}-$DATASET-Seed${SEED}

if [[ $SCALE > 2 && "$PRE_TRAIN" == "true" ]] ; then
    PRE_TRAIN_ARG="--pre_train ../$EXPERIMENT_DIR/${TEMPLATE}-X2-$DATASET-Seed${SEED}/model/model_best.pt"
else
    PRE_TRAIN_ARG="" 
fi


python main.py \
    --experiment_dir $EXPERIMENT_DIR \
    --load $MODEL_DIR \
    --save $MODEL_DIR \
    $PRE_TRAIN_ARG \
    --n_threads 0 \
    --seed $SEED \
    --tensorboard \
    \
    --template $TEMPLATE \
    --scale $SCALE \
    --shift_mean False \
    \
    --cache_data \
    --dir_data ../../Data \
    --data_train $DATASET \
    --data_test $DATASET \
    --data_range $TRAIN_DATA_RANGE \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --n_colors 1 \
    \
    --print_every 50 \
    --test_every $TEST_EVERY \
    --epochs $EPOCHS \
