import os
from config_global import teeth_class, UNET_TEETH_EPOCH, UNET_TEETH_BATCH, MODEL_PATH_UNET_TEETH


class UNetConfig:

    def __init__(self,
                 epochs = UNET_TEETH_EPOCH,  # Number of epochs
                 batch_size = UNET_TEETH_BATCH,    # Batch size
                 validation = 10.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold = 0.5,
                 optimizer='Adam',
                 lr = 0.0001,     # learning rate
                 lr_decay_milestones = [20, 50],
                 lr_decay_gamma = 0.9,
                 weight_decay=1e-8,
                 momentum=0.9,
                 nesterov=True,

                 n_channels = 3, # Number of channels in input images
                 n_classes = len(teeth_class)+1,  # Number of classes in the segmentation
                 scale = 1,    # Downscaling factor of the images
                 load = False,   # Load model from a .pth file
                 save_cp = True,
                 model='UNet',
                 bilinear = False,
                 deepsupervision = False,
                 ):
        super(UNetConfig, self).__init__()

        self.checkpoints_dir = MODEL_PATH_UNET_TEETH
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
