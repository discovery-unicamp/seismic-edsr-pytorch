# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program execution
# - model
# - data
# - training
# - optimizer

#    --load toy_x2 \
#    --save_results \
#    --resume -1 \
#    --test_only \
#    --reset \
#    --debug \

SCALE=2
DATASET=NAMSS
PATCH_SIZE=$(( 48*SCALE ))
DATA_RANGE='1-2746/1-350'

# Obs: the resume=0 and pre_train arguments allows to test with the best model instead of the latest
    #--load edsrX${SCALE}_$DATASET \
    #--save edsrX${SCALE}_$DATASET \
    #--resume 0 \
    #--pre_train "../experiment/edsrX${SCALE}_$DATASET/model/model_best.pt" \
python main.py \
    --test_only \
    \
    --template EDSR_paper \
    --scale $SCALE \
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
