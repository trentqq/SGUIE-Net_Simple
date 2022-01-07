import torch
from torchvision.models import vgg
from .base_model import BaseModel
from . import networks
import random
import numpy as np
import cv2


class SGUIENetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(netS_norm='batch', netS='unet_256')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(no_dropout=True)

        return parser

    def __init__(self, opt):
        """Initialize the SCGUICE class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['pixel']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['raw', 'gt_mask', 'pred_enhancement', 'ref_enhancement']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']

        else:  # during test time, load netG
            self.model_names = ['G']
        # define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.netG_norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # load VGG
            # self.vgg = vgg.vgg16().cuda()
            # state_dict = torch.load(opt.vgg_pretrained_path)
            # self.vgg.load_state_dict(state_dict)
            # self.vgg.eval()
            # self.set_requires_grad(self.vgg, False)

            # define loss functions
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # self.colorLoss = networks.ColorLoss()
            #self.ssimLoss = networks.StructureConsistencyLoss()
            # self.pccLoss = networks.PCC(opt.pcc_patch, opt.pcc_thresh)
            #self.tvLoss = networks.TVLoss()
            #self.contrastSaturationLoss = networks.ContrastSaturationLoss(opt.wx)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # self.optimizers.append(self.optimizer_S)
            self.optimizers.append(self.optimizer_G)

    def split_by_semantic(self):
        """
        Split input images based on semantic information
        :return: semantic regions
        """
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
        """

        self.raw = input['raw'].to(self.device)
        self.ref_enhancement = input['ref'].to(self.device)
        self.gt_mask = input['mask'].to(self.device)
        self.image_paths = input['raw_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.pred_mask = self.gt_mask
        self.regions, self.region_masks, self.region_idx = self.split_by_semantic()
        self.pred_enhancement, self.pred_region_imgs = self.netG(self.raw, self.regions, self.region_masks, self.region_idx)  # G(A)


    def backward_G(self):
        """Calculate the loss for generator
        - GAN loss
        """
        # l2 loss
        self.loss_pixel = self.criterionL2(self.pred_enhancement, self.ref_enhancement) * 10

        # region loss
        for i, region_idx in enumerate(self.region_idx):
            (x, y, w, h) = region_idx
            ref_region = self.ref_enhancement[:, :, y:y+h, x:x+w]
            self.loss_pixel += self.criterionL2(self.pred_region_imgs[i], ref_region) * 10

        # total loss
        self.loss_G = self.loss_pixel
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

