import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2


class TestSGUIEModel(BaseModel):
    """ This class implements the Test SGUIE-Net model
    The model training requires '--dataset_mode single' dataset.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(netG_input_nc=3)

        return parser

    def __init__(self, opt):
        """Initialize the SCGUICE class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['pred_enhancement']
        self.model_names = ['G' + opt.model_suffix]
        self.netG = networks.define_G(opt.netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.netG_norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def split_by_semantic(self):
        regions = []
        region_masks = []
        region_idx = []
        masks_np = (self.gt_mask * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        for i in range(masks_np.shape[2]):
            mask = np.array(masks_np[:, :, i])
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = []
            # 所有相同语义区域的组成的外接矩形
            for cont in contours:
                contour.extend(cont)
            if len(contours) == 0:
                continue
            rect = cv2.boundingRect(np.array(contour))
            (x, y, w, h) = rect
            if w < 32 or h < 32:
                continue
            regions.append(self.raw[:, :, y:y+h, x:x+w].clone())
            region_masks.append(torch.unsqueeze(self.gt_mask[:, i, y:y + h, x:x + w], 1))
            region_idx.append(rect)
        return regions, region_masks, region_idx


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        self.raw = input['raw'].to(self.device)
        self.gt_mask = input['mask'].to(self.device)
        self.image_paths = input['raw_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_mask = self.gt_mask
        self.regions, self.region_masks, self.region_idx = self.split_by_semantic()
        self.pred_enhancement, _ = self.netG(self.raw, self.regions, self.region_masks, self.region_idx)  # G(A)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

