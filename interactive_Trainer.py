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
import torchvision.transforms.functional as F

# import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
# from . import Hook
from . import my_utils as Mutils
import matplotlib.pyplot as plt          
import pandas as pd
# from . import metrics_base as MB
class interactive_nnUNetTrainer(nnUNetTrainer):
    """custom nnUNet Trainer that train also for interactive segmentation and prediction refinement"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
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
            unpack_dataset,
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
        self.num_epochs = 400 #change this to change max number of epoch 

    def _build_loss(self,):
        """function to define the loss of the model and to configure the deep supervision"""

        loss=Mutils.loss_P0_and_click_region({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'ignore_index':self.label_manager.ignore_label}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        
        if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = Mutils.DeepSupervisionWrapper(loss, weights)
        return loss

    def add_guidance(self,data,target,training_mode=True,mode='global'):
        return Mutils.click_simulation_test(self,data,target,training_mode,click_mode=mode)
    
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if self.current_epoch>0:
            with torch.no_grad():
                # data=self.add_guidance(data,target,'global')
                data[:,1:]=data[:,1:]*0
                net_output0=self.network(data)
                self.loss.loss.net_output0=net_output0
                data,click_map=self.add_guidance(data,target,'global')
        
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            # l = self.loss(output, target)
            if self.current_epoch>0:
                self.loss.click_map=torch.sum(torch.where(click_map>0,1,0),axis=1)
                self.loss.loss.alpha=1-(self.current_epoch/self.num_epochs)
                # self.loss.loss.alpha = np.exp(-5*(self.current_epoch/self.num_epochs))
            
            l=self.loss(output,target)
           
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}



    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            data[:,1:]=data[:,1:]*0
            net_output0=self.network(data)
            self.loss.loss.net_output0=net_output0
            data,click_map=self.add_guidance(data,target,training_mode=False)
            
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data

            self.loss.click_map=torch.sum(torch.where(click_map>0,1,0),axis=1)
            self.loss.loss.alpha=1-(self.current_epoch/self.num_epochs)
            # self.loss.loss.alpha = np.exp(-5*(self.current_epoch/self.num_epochs))
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
