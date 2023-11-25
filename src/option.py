import argparse
import template
import torch
import sys

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# TensorBoard
parser.add_argument('--tensorboard', action='store_true',
                    help='Uses TensorBoard to visualize training.')
parser.add_argument('--tensorboard_keep_alive', action='store_true',
                    help='Do not exit after training, keeps program running.')
parser.add_argument('--tensorboard_nimgs', type=int, default=2,
                    help='Number of images to plot in TensorBoard during test.')
parser.add_argument('--tensorboard_color', type=str, default='gray',
                    help='Color to plot images in TensorBoard.')
parser.add_argument('--tensorboard_dir', type=str, default='tensorboard',
                    help='Directory for TensorBoard logs.')
parser.add_argument('--tensorboard_port', type=str, default='6006',
                    help='Port to run TensorBoard.')
parser.add_argument('--tensorboard_host', type=str, default='0.0.0.0',
                    help='Host to run TensorBoard.')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--cache_data', action='store_true',
                    help='Cache data to RAM.')
parser.add_argument('--input_range', type=str, default='0,255',
                    help=('Minimum and maximum possible values of the input signal,'
                    'separated by a comma. Ex: RGB range is typically "0,255".'))
parser.add_argument('--tensor_range', type=str, default='-1,1',
                    help=('New range to linearly transform the input before it goes into the network.'))
parser.add_argument('--dir_data', type=str, default='../../../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--max_chop_slices', type=int, default=6,
                    help='max number of slices per dimension to chop an image')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Deep Back-Projection Networks (DBPN)
parser.add_argument('--n0', type=int, default=256,
                    help='DDBPN: number of filters used in the initial LR features extraction.')
parser.add_argument('--nr', type=int, default=64,
                    help='DDBPN: number of filters used in each projection unit.')
parser.add_argument('--T', type=int, default=7,
                    help='DDBPN: number of stages (up and down projection units) in the network.')

# Option for Residual Dense Network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for WDSR
parser.add_argument('--r_mean', type=float, default=0.4488,
                    help='Mean of R Channel')
parser.add_argument('--g_mean', type=float, default=0.4371,
                    help='Mean of G channel')
parser.add_argument('--b_mean', type=float, default=0.4040,
                    help='Mean of B channel')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_patience', type=int, default=15,
                    help='Patience (in epochs) do update learning rate')
parser.add_argument('--lr_max_updates', type=int, default=6,
                    help='Max. number of times lr is updated.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Log specifications
parser.add_argument('--experiment_dir', type=str, default='experiment',
                    help='base experiments directory. Default: experiment')
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--benchmark', action='store_true',
                    help='use to differentiate between valid and test data')

args = parser.parse_args()

if args.cache_data and args.n_threads != 0:
    import sys
    sys.exit("When using --cache_data, --n_threds must be set to 0 to avoid multiple copies of the whole dataset being loaded to RAM. Exiting.")

if args.benchmark and not args.test_only:
    import sys
    sys.exit("--benchmark option should only be used in --test_only mode. Exiting.")

# No need to chache data that will be read only once
if args.test_only:
    args.cache_data = False

# Set CPU to True if no GPU available
if not torch.cuda.is_available():
    args.cpu = True

# Automatically set all available GPUs, unless --n_GPUs was explicitly set
if not args.cpu:
    if not '--n_GPUs' in sys.argv:
        args.n_GPUs = torch.cuda.device_count()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.input_range = [float(x) for x in args.input_range.split(',')]
args.tensor_range = [float(x) for x in args.tensor_range.split(',')]
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

# Maybe latter change code that needs this value
args.rgb_range = args.input_range[1] - args.input_range[0]

template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

