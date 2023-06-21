
def set_template(args):
    if args.template.lower().find('toy') >= 0:
        # Model
        args.model = 'TOY'
        args.act = 'relu'
        args.n_resblocks = 32
        args.n_feats = 32
        args.res_scale = 1
        args.shift_mean = False

        # Training
        args.loss = '1*L1'
        args.batch_size = 16

        # Optimizer
        args.optimizer = 'ADAM'
        args.lr = 0.0001
        args.gamma = 0.5
        args.betas = (0.9, 0.999)

    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        #args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        # Model
        args.model = 'EDSR'
        args.act = 'relu'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

        # Training
        args.loss = '1*L1'
        args.batch_size = 16
        args.patch_size = 48 * args.scale[-1]

        # Optimizer
        args.optimizer = 'ADAM'
        args.betas = (0.9, 0.999)
        args.epsilon = 1e-8

        args.lr = 0.0001
        args.lr_patience = 15
        args.lr_max_updates = 6
        args.gamma = 0.5

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        # Model
        args.model = 'DDBPN'
        args.n0 = 256
        args.nr = 64
        args.T = 7
        args.chop = True

        # Training
        args.loss = '1*L1'
        args.batch_size = 16
        args.patch_size = 40 * args.scale[-1]

        ## Optimizer
        args.optimizer = 'ADAM'
        args.betas = (0.9, 0.999)
        args.epsilon = 1e-8
        args.weight_decay = 0

        args.lr = 0.0001
        args.lr_patience = 15
        args.lr_max_updates = 6
        args.gamma = 0.5

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        args.lr = 5e-5
        #args.decay = '150'

    if args.template.find('RDN') >= 0:
        # Model
        args.model = 'RDN'
        args.G0 = 64
        args.RDNkSize = 3
        args.RDNconfig = 'B'

        # Training
        args.loss = '1*L1'
        args.batch_size = 16
        args.patch_size = 32 * args.scale[-1]

        ## Optimizer
        args.optimizer = 'ADAM'
        args.betas = (0.9, 0.999)
        args.epsilon = 1e-8

        args.lr = 0.0001
        args.lr_patience = 15
        args.lr_max_updates = 6
        args.gamma = 0.5

    if args.template.find('WDSR') >= 0:
        # Model
        args.model = 'WDSR'
        args.n_feats = 64
        args.n_resblocks = 16
        args.res_scale = 1

        # Training
        args.loss = '1*L1'
        args.batch_size = 16
        args.patch_size = 48 * args.scale[-1]

        ## Optimizer
        args.optimizer = 'ADAM'
        args.betas = (0.9, 0.999)
        args.epsilon = 1e-8

        args.lr = 0.001
        args.lr_patience = 15
        args.lr_max_updates = 6
        args.gamma = 0.5

    if args.template.find('RCAN') >= 0:
        # Model
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.reduction = 16
        args.res_scale = 1
        #args.chop = True

        # Training
        args.loss = '1*L1'
        args.batch_size = 16
        args.patch_size = 48 * args.scale[-1]

        # Optimizer
        args.optimizer = 'ADAM'
        args.betas = (0.9, 0.999)
        args.epsilon = 1e-8

        args.lr = 0.0001
        args.lr_patience = 20
        args.lr_max_updates = 6
        args.gamma = 0.5

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

