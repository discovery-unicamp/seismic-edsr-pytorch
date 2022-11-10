# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program exectution
# - model
# - data
# - training
# - optimizer

#    --load toy_x2 \
#    --save_results \
#    --test_only \
#    --reset \
#    --pre_train ../experiment/toy_tiff_X2/model/model_best.pt \
#    --tensorboard \

#python -m pdb main.py \
python main.py \
    --reset \
    --resume -1 \
    --tensorboard \
    --port_tensorboard 6006 \
    --template TOY \
    --n_threads        12 \
    --seed             42 \
    --load toy_namss_2x \
    --save toy_namss_2x \
    \
    --scale '2' \
    \
    --lr_patience 2 \
    --lr_max_updates 6 \
    \
    --print_every 2 \
    --test_every 20 \
    --epochs 10 \
