import numpy as np
import torch
import torchvision.transforms.functional as F
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

# from . import Hook
from . import my_utils as Mutils
import matplotlib.pyplot as plt          
import pandas as pd
# from . import metrics_base as MB
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform,
    RenameTransform,
    NumpyToTensor,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    save_json,
    maybe_mkdir_p,
)
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    resample_and_save,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import (
    MoveSegAsOneHotToData,
    ApplyRandomBinaryOperatorTransform,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)
# from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import (
#     LimitedLenWrapper,
# )
# from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
# from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import (
#     ConvertSegmentationToRegionsTransform,
# )
# from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import (
#     Convert2DTo3DTransform,
#     Convert3DTo2DTransform,
# )
# from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
# from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
# from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
# from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
# from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
# from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
# from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
# from nnunetv2.utilities.collate_outputs import collate_outputs
# from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
# from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
# from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
# from nnunetv2.utilities.helpers import empty_cache, dummy_context
# from nnunetv2.utilities.label_handling.label_handling import (
#     convert_labelmap_to_one_hot,
#     determine_num_input_channels,
# )
# from nnunetv2.utilities.plans_handling.plans_handler import (
#     PlansManager,
#     ConfigurationManager,
# )
from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from . import nnUNetTrainer_interactive

def binary_to_decimal(image,):
        """function to convert binary click map into decimal one. needed to compute our specific loss function with click region bias
        image shape should be (b,c,d,h,w)"""
        nchan = image.shape[1]
        powers = (1 << np.arange(nchan)).astype(image.dtype)  
        return np.tensordot(powers, image, axes=(0, 0))

class binary_Trainer(nnUNetTrainer_interactive.nnUNetTrainerinteractive):
    """custom nnUNet Trainer that train also for interactive segmentation and prediction refinement"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        # max_iter=5,
        # nbr_supervised=0.5,
    ):
        """Initialize the nnU-Net trainer for muscle segmentation."""
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device,
        )
        self.max_iter = (
            5  # max_iter # maximal number of click during an iteration of training
        )
        self.nbr_supervised = (
            0.75  # nbr_upervised  # number of image that are trained with clicks
        )
        self.dataset_json = dataset_json  # info on the dataset -> maybe not useful
        
        self.incr = 0
        self.epoch_incr = 0
        self.enable_deep_supervision = True
        self.num_epochs = 800 #change this to change max number of epoch 
        self.N_alpha = 800

    def add_guidance(self,data,target,training_mode=True,mode='global'):
        return Mutils.click_simulation_binary(self,data,target,training_mode,click_mode=mode)
    
