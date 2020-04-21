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
#    --resume -1 \
#    --test_only \
    #--reset \
# python -m pdb main.py \

python main.py \
    --resume -1 \
    --load edsr_tiff_X2 \
    --tensorboard \
    --template edsr_tiff \
    --n_threads        6 \
    --seed             42 \
    --save edsr_tiff_X2 \
    \
    --scale '2' \
    \
    --print_every 50 \
    --test_every 1000 \
    --epochs 110 \
