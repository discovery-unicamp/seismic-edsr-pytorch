
def set_template(args):
    # My templates
    if args.template.lower().find('toy') >= 0:
        import sys
        args.ext = 'img'

        # If cpu was not explicitly set,
        # set if available
        if not '--cpu' in sys.argv:
            from torch import cuda
            if cuda.is_available():
                args.cpu = False
            else:
                args.cpu = True

        # Model
        args.model = 'TOY'
        args.act = 'relu'
        args.n_resblocks = 32
        args.n_feats = 32
        args.res_scale = 1
        args.shift_mean = False

        # Data
        args.dir_data = '../../Data'
        args.data_train = 'DIV2K'
        args.data_test = 'DIV2K'
        #args.data_range = '1-15/16-20'
        args.data_range = '1-1/16-16'
        args.patch_size = 48
        args.rgb_range = 255
        args.n_colors = 3

        # Training
        args.loss = '1*L1'
        args.batch_size = 16

        # Optimizer
        args.optimizer = 'ADAM'
        args.lr = 0.0001
        args.momentum = 0.9
        args.decay = '200'
        args.gamma = 0.5
        args.weight_decay = 0
        args.betas = (0.9, 0.999)

    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

