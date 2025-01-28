import torch
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import pandas as pd 
import time
# from . import metrics_base as MB

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import SoftDiceLoss
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from scipy.ndimage import distance_transform_cdt

def chamfer_distance_torch(T1,T2):
    """return the tensor containing the distance of from T2 of each point of T1 """
    return torch.min(torch.cdist(T1,T2),dim=1)[0]

def from_flat_to_shaped_idx(flat_idx,original_shape):
    D,H,W=original_shape
    d_idx=flat_idx // (H*W)
    rest=flat_idx % (H*W)
    h_idx=rest // W
    w_idx=rest % W
    return torch.tensor([d_idx,h_idx,w_idx])

def scipy_get_one_click(gt,seg,ignore_label=None):
    """ compute chamfer distance via scipy fct then return a click and its associated label"""
    errors=(gt!=seg).float()
    chamfer_distance=torch.tensor(distance_transform_cdt(errors.cpu(),metric='taxicab'),device=(torch.device('cuda:0')))
    if ignore_label is not None :
        ignore_mask=gt!=ignore_label
    if ignore_mask.sum()!=0:
        chamfer_distance=chamfer_distance*ignore_mask
    proba=torch.exp(chamfer_distance)-1
    proba_with_treshold=torch.where(proba>2,proba,0)    
    if proba_with_treshold.sum()==0:
        return None,None
    try:
        random_pick = torch.multinomial(proba_with_treshold.flatten(),1,replacement=True) #normalisation is done by the torch function
    except:
        breakpoint()
             
    click=from_flat_to_shaped_idx(random_pick,proba_with_treshold.shape)
    chosen_label=gt[click[0],click[1],click[2]].int().item()

    return click,chosen_label

                                
def dilatation(image,device=None):
    """dilate the "1" part of an 2d binary image"""
    kernel=torch.tensor([[1,1,1],[1,1,1],[1,1,1]],device=device).unsqueeze(0).unsqueeze(0).float()
    if device ==None:
        device=("cuda" if torch.cuda.is_available() else "cpu")
    return torch.where(torch.conv2d(image.unsqueeze(0).unsqueeze(0),kernel,padding=1)>=1,1,0).squeeze(0).squeeze(0)

def dilatation_in_3d(image,device=None):
    """3d dilatation for an 3d binary image """
    image=image.float()
    image_type=image.dtype
    if device ==None:
        device=("cuda" if torch.cuda.is_available() else "cpu")
    kernel=torch.full((3,3,3),1,dtype=image_type,device=device).unsqueeze(0).unsqueeze(0)
    return torch.where(TF.conv3d(image.unsqueeze(0).unsqueeze(0),kernel,padding=1)>=1,1,0).squeeze(0).squeeze(0)

def erosion_3d(image,ignore_map=None):
    if ignore_map is None:
        return 1-dilatation_in_3d(1-image)
    else:
        background=(1-image)*ignore_map
        background_dilatation=dilatation_in_3d(background)
        return (1-background_dilatation)*ignore_map
    
def chamfer_dist_morpho(image,ignore_map=None,iteration=20):
    D={0:image}
    results=torch.zeros(image.shape,device=image.device)
    for k in range(0,iteration):
        D[k+1]=erosion_3d(D[k],ignore_map)
        if D[k+1].sum()==0:
            return results
        results+=D[k+1]
    return results

def get_one_click_morpho(gt,seg,ignore_label=None):
    errors = (gt!=seg).float()
    if ignore_label is not None:
        is_ignored = torch.any(gt==ignore_label,axis=[1,2])
        is_seg = 1 - (is_ignored.int().unsqueeze(1).unsqueeze(1))
    with torch.cuda.amp.autocast():
        chamfer_mask = chamfer_dist_morpho(errors,is_seg)
        proba = torch.exp(chamfer_mask)-1
        if torch.amax(proba) > 0:
            click = torch.multinomial(proba.flatten(),1,replacement=True)
            click = from_flat_to_shaped_idx(click,proba.shape)
            chosen_label = gt[click[0],click[1],click[2]].int().item()
            return click, chosen_label
        return None,None

def get_probabilities(mask,ignore_labelmap,mode='3d'):
    """mask is the error map, ignore_labelmap the binary map of pixels that are not labelized ignore"""
    if ignore_labelmap.sum()!=0:
        mask=mask*ignore_labelmap
    if mode=='3d':
        contour=dilatation_in_3d(mask)-mask
    else:
        contour=dilatation(mask)-mask

    if ignore_labelmap.sum()!=0:
       contour=contour*ignore_labelmap

    dist=chamfer_distance_torch(torch.nonzero(mask==1).float(),torch.nonzero(contour==1).float())
    return torch.exp(dist)-1

def do_simulate(k,N,training_mode):
            """training_mode allow us to use this function differently when we are in the validation part 
            where we want to provide the same number of click everytime"""
            if training_mode:
                return np.random.binomial(n=1,p=1-k/N)
            else:
                return True

def click_simulation(self,data,target,training_mode=True):
    """ main function that allows us to simulate clicks
    training_mode : True if you use this function during training, False if it's during validation"""
    b,c,d,h,w= data.size()  # batch, channel, depth, height, width
    groundtruth=target[0]
    #get the number of labels

    if 'ignore_label' in self.dataset_json['labels'].keys():
        nbr_labels=len(self.dataset_json['labels'])-1 #-1 because we don't count the ignore label
    else:
        nbr_labels=len(self.dataset_json['labels'])

    #click channel are potentialy noised by the preprocessing we need to fix that : 
    if data[:,1:nbr_labels+1].sum()!=0:
        data[:,1:nbr_labels+1]=torch.full((b,c-1,d,h,w),0,device=self.device,dtype=torch.int)

    click_mask= torch.full((b, c - 1,d, h, w), 0, device=self.device, dtype=torch.float)

    if np.random.binomial(n=1,p=self.nbr_supervised): #choosing if the batch is gonna be train with clicks or not   
        self.network.eval() #putting the model in inference mode, needed to simulate click 
        for k in range(self.max_iter):
        #we first want to get map probabilities
            if do_simulate(k,self.max_iter,training_mode):
                # using current network to have prediction & probabilities 
                
                with torch.no_grad():
                    data[:,1:]=apply_gaussian_bluring(click_mask,(1,2,2),5)
                    logits = self.network(data)
                    probabilities = torch.softmax(logits[0],dim=1)
                    _,prediction = torch.max(probabilities,dim=1)
                for nimage in range(b):
                    
                    test = groundtruth[nimage][0] == prediction[nimage] #test matrix to find prediction's mistakes
                    misslabeled_indices = torch.nonzero(~test) #getting indexes of misslabeled pixels
                    
                    for slice in range(d):

                        #checking if the current slice is fully blank
                        try:
                            if torch.nonzero(data[nimage,0,slice]-data[nimage,0,slice,0,0]).numel()==0:
                                print('slice fully blank')
                                continue
                        except:
                            breakpoint()
                            

                        #filtering by slice                        
                        mask=(misslabeled_indices[:,0]==slice)
                        misslabeled_per_slice=misslabeled_indices[mask]

                        #getting gt value of all wrongly predicted pixels
                        slice_indices=misslabeled_per_slice[:,0]
                        h_indices=misslabeled_per_slice[:,1]
                        w_indices=misslabeled_per_slice[:,2]
                        true_value=groundtruth[nimage,0][slice_indices,h_indices,w_indices]

                        #getting the worst prediction label
                        label_list,misslabeled_count=true_value.unique(return_counts=True)
                        if len(misslabeled_count)==0: #when the network does 0 mistake 
                            print('no misslabeled pixel (perfect prediction)')
                            continue
                        else:
                            max_value=torch.max(misslabeled_count).item()
                            worst_labels=(misslabeled_count==max_value).nonzero(as_tuple=False)
                            chosen_label=int(label_list[worst_labels][np.random.randint(len(worst_labels))].item())
        
                            #simulation du clique ici:
                            mask=(groundtruth[nimage,0,slice]==chosen_label) & (~test[slice])
                            # potential_click = torch.nonzero(mask)
                            
                            #choose click with respect to probabilites from chamfer distance                                                                               
                            gt_label=(groundtruth[nimage,0,slice]==chosen_label).int()
                            pred_label=(prediction[nimage,slice]==chosen_label).int()
                            False_negative=torch.nonzero(torch.where(gt_label-pred_label==1,1,0))
                            D_plus=torch.full((h,w),0,device=self.device).float()
                            h_indices=False_negative[:,0]
                            w_indices=False_negative[:,1]
                            D_plus[h_indices,w_indices]=1
                            
                            if 1:#D_plus.sum()>=D_minus.sum():
                            
                                proba_click=get_probabilities(D_plus)
                                proba_with_treshold=torch.where(proba_click>2,proba_click,0)
                                try:
                                    random_pick = torch.multinomial(proba_with_treshold,1,replacement=True) #normalisation is done by the torch function
                                except:
                                    continue
                                click=torch.nonzero(D_plus)[random_pick].squeeze(0)
                            
                            del False_negative, proba_click
                            #adding click into data
                            
                            click_mask[nimage,chosen_label,slice,click[0],click[1]] = 1                                 
                                 
            else:
                break
        # here we smoothed the click data
        data[:,1:]=apply_gaussian_bluring(click_mask,2,5)
        if training_mode:
            self.network.train() #putting the model back to training mode 
        print('all click generated!, starting gradient descent...')
    return data

def click_simulation_2d(self,data,target,training_mode=True,factor=1000):

    # Part where we are going to simulate clicks:
    # starting by creating channels to store clicks:
    b, c, h, w = data.size()  # batch, channel, depth, height, width
    groundtruth = target[0]

    # get the number of labels

    if "ignore_label" in self.dataset_json["labels"].keys():
        nbr_labels = (
            len(self.dataset_json["labels"]) - 1
        )  # -1 because we don't count the ignore label
    else:
        nbr_labels = len(self.dataset_json["labels"])

    # click channel are potentialy noised by the preprocessing we need to fix that :
    if data[:, 1 : nbr_labels + 1].sum() != 0:
        data[:, 1 : nbr_labels + 1] = torch.full(
            (b, c - 1, h, w), 0, device=self.device, dtype=torch.int
        )
    click_mask= torch.full((b, c - 1, h, w), 0, device=self.device, dtype=torch.int)
    # function to decide if we simulate the k-th click
    if np.random.binomial(
        n=1, p=self.nbr_supervised
    ):  # choosing if the batch is gonna be train with clicks or not
        self.network.eval()  # putting the model in inference mode, needed to simulate click
        # for k in range(self.max_iter):
        for k in range(self.max_iter):
            # we first want to get map probabilities
            if do_simulate(k, self.max_iter,training_mode):
                # using current network to have prediction & probabilities

                with torch.no_grad():
                    data[:,1:]=click_mask
                    data=blurred_data(data,(5,5),(2,2),factor=factor)
                    logits = self.network(data)
                    probabilities = torch.softmax(logits[0], dim=1)
                    _, prediction = torch.max(probabilities, dim=1)
                
                    
                for nimage in range(b):

                    test = (
                        groundtruth[nimage][0] == prediction[nimage]
                    )  # test matrix to find prediction's mistakes
                    misslabeled_indices = torch.nonzero(
                        ~test
                    )  # getting indexes of misslabeled pixels

                    # checking if the current slice is fully blank
                    try:
                        if (
                            torch.nonzero(
                                data[nimage, 0] - data[nimage, 0, 0, 0]
                            ).numel()
                            == 0
                        ):
                            # print('slice fully blank')
                            continue
                    except:
                        breakpoint()

                    # getting gt value of all wrongly predicted pixels

                    h_indices = misslabeled_indices[:, 0]
                    w_indices = misslabeled_indices[:, 1]
                    true_value = groundtruth[nimage, 0][h_indices, w_indices]

                    # getting the worst prediction label

                    label_list, misslabeled_count = true_value.unique(
                        return_counts=True
                    )
                    if (
                        len(misslabeled_count) == 0
                    ):  # when the network does 0 mistake
                        continue
                    else:
                        proportional_count = torch.tensor(
                            [
                                misslabeled_count[k]
                                / (groundtruth[nimage, 0] == label_list[k]).sum()
                                for k in range(len(misslabeled_count))
                            ],
                            device=self.device,
                        )
                        # max_value=torch.max(misslabeled_count).item()
                        # worst_labels=(misslabeled_count==max_value).nonzero(as_tuple=False)
                        max_value = torch.max(proportional_count).item()
                        worst_labels = (proportional_count == max_value).nonzero(
                            as_tuple=False
                        )
                        chosen_label=int(label_list[worst_labels][np.random.randint(len(worst_labels))].item())
                       
                        # simulation du clique ici
                                                
                        # choose click with respect to probabilites from chamfer distance

                        gt_label = (groundtruth[nimage, 0] == chosen_label).int()
                        pred_label = (prediction[nimage] == chosen_label).int()
                        False_negative = torch.nonzero(
                            torch.where(gt_label - pred_label == 1, 1, 0)
                        )
                        D_plus = torch.full((h, w), 0, device=self.device).float()
                        h_indices = False_negative[:, 0]
                        w_indices = False_negative[:, 1]
                        D_plus[h_indices, w_indices] = 1
                        if True:  # D_plus.sum()>=D_minus.sum():
                            # breakpoint()
                            proba_click = get_probabilities(D_plus)
                            proba_with_treshold=torch.where(proba_click>2,proba_click,0)
                            try:
                                random_pick = torch.multinomial(proba_with_treshold,1,replacement=True) #normalisation is done by the torch function             

                            except:
                                continue
                            click = torch.nonzero(D_plus)[random_pick].squeeze(0)
                            del False_negative, proba_click
                            
                            # adding click into data
                            try:
                                click_mask[
                                    nimage, chosen_label, click[0], click[1]
                                ] = 1

                            except:
                                print('AU MOMENT DE METTRE LE CLICK DANS LE TENSEUR!!!\n')
                                print(f"click:{click}")
                                breakpoint()

            else:
                break
        # here we smoothed the click data
        data[:,1:]=click_mask
        data=blurred_data(data,(5,5),(2,2))
        if training_mode:
            self.network.train()  # putting the model back to training mode

    return data 

def blurred_data(data,kernel_size,sigma,factor=1000):
    """ use only for 2D ! """
    if len(data.shape)==4:
        n,c,h,w=data.shape
    if len(data.shape)==5:
        n,c,d,h,w=data.shape
    for channel in range(1, c):
        data[:, channel] = F.gaussian_blur(data[:, channel], kernel_size, sigma)*factor
    return data       

def gaussian_kernel_3d(size, sigma):
    """create a gaussian kernel with Pytorch, designed for a 3d usage
    sigma can be int (isotropic model) or 3d vector
    size is a int for the kernel size -> (size,size,size)"""

    if type(sigma)==int:
        sigma=[sigma]*3
    kernel_range = torch.arange(-size//2 + 1, size//2 + 1,device=torch.device('cuda'))
    x, y, z = torch.meshgrid(kernel_range, kernel_range, kernel_range, indexing='ij')
    kernel = torch.exp(-(x**2/(2. * sigma[0]**2) + y**2/(2. * sigma[1]**2) + z**2/(2. * sigma[2]**2)))
    kernel /= torch.sum(kernel)  # Normalization
    return kernel

def apply_gaussian_bluring(input,sigma,kernel_size,factor=1000):
    """input is all the clicks channels,
    sigma can be int or tuple
    kernel_size int """
    n,c,d,h,w=input.shape
    kernel=gaussian_kernel_3d(kernel_size,sigma)
    input = input.reshape(n*c,1,*input.shape[-3:])
    kernel = kernel.reshape(1, 1, *kernel.shape)
    output= TF.conv3d(input,kernel,stride=1,padding='same')
    return output.reshape(n,c,*output.shape[-3:])*factor
 
    
def alternative_click_simulation_2d(self,data,target,training_mode=True,factor=1000):

    # Part where we are going to simulate clicks:
    # starting by creating channels to store clicks:
    b, c, h, w = data.size()  # batch, channel, depth, height, width
    groundtruth = target[0]

    # get the number of labels

    click_mask= torch.full((b, c - 1, h, w), 0, device=self.device, dtype=torch.int)
    # function to decide if we simulate the k-th click
    if np.random.binomial(
        n=1, p=self.nbr_supervised
    ):  # choosing if the batch is gonna be train with clicks or not
        self.network.eval()  # putting the model in inference mode, needed to simulate click
        # for k in range(self.max_iter):
        for k in range(self.max_iter):
            # we first want to get map probabilities
            if do_simulate(k, self.max_iter,training_mode):
                    # using current network to have prediction & probabilities

                with torch.no_grad():
                    data[:,1:]=click_mask
                    data=blurred_data(data,(5,5),(2,2),factor=factor)
                    logits = self.network(data)
                    probabilities = torch.softmax(logits[0], dim=1)
                    _, prediction = torch.max(probabilities, dim=1)
                    test=prediction!=groundtruth.squeeze(1)

                for nimage in range(b):

                    # checking if the current slice is fully blank
                    try:
                        if (
                            torch.nonzero(
                                data[nimage, 0] - data[nimage, 0, 0, 0]
                            ).numel()
                            == 0
                        ):
                            # print('slice fully blank')
                            continue
                    except:
                        breakpoint()
                    #getting all misslabelled pixel 

                    ERRORS=test[nimage].float()
                    if ERRORS.sum()==0:
                        print('no error on this image')
                        continue

                    proba_click=get_probabilities(ERRORS)

                    proba_with_treshold=torch.where(proba_click>2,proba_click,0)

                    try:
                        random_pick = torch.multinomial(proba_with_treshold,1,replacement=True) #normalisation is done by the torch function             
                    except:
                        print('no click generated because there is only thin mistakes')
                        continue
                    click=torch.nonzero(ERRORS)[random_pick].squeeze(0)
                    click_label=groundtruth[nimage,0][click[0],click[1]].int()
                        
                        # adding click into data
                    try:
                        click_mask[
                            nimage, click_label, click[0], click[1]
                        ] = 1

                    except:
                        print('AU MOMENT DE METTRE LE CLICK DANS LE TENSEUR!!!\n')
                        print(f"click:{click}")
                        breakpoint()
                    # if training_mode:
                        
                    #     self.report.loc[len(self.report)]={'click_x': click[0].item(),
                    #                                                     'click_y': click[1].item(),
                    #                                                     'batch': self.batch_incr,
                    #                                                     'epoch': self.current_epoch,
                    #                                                     'nimage': nimage,
                    #                                                     'label': click_label.item()}|MB.compute_dice_per_label(groundtruth[nimage,0],prediction[nimage],9)

            else:
                break
        # here we smoothed the click data
        data[:,1:]=click_mask
        data=blurred_data(data,(5,5),(2,2))
        if training_mode:
            self.network.train()  # putting the model back to training mode

    return data 

def choose_label(gt,seg,click_rule='proportional'):
        """ function to choose the label we are going to correct"""
        test= gt!=seg
        FN_mask=torch.where(test==1,gt,torch.nan)
        #proportional way
        if click_rule=='proportional':
            misslabeled_labels,misslabeled_values=torch.unique(FN_mask[~torch.isnan(FN_mask)],return_counts=True)
            if misslabeled_values.sum()==0:
                return None
            proportional_misslabeled=torch.tensor([misslabeled_values[k]/(gt==misslabeled_labels[k]).sum() for k in range(len(misslabeled_labels))],device=torch.device('cuda:0'))
            potential_choice=misslabeled_labels[proportional_misslabeled==torch.max(proportional_misslabeled).item()]
            chosen_label=potential_choice[torch.randint(len(potential_choice), (1,))]
        else:
        #max way
            misslabeled_labels,misslabeled_values=torch.unique(FN_mask[~torch.isnan(FN_mask)],return_counts=True)
            if misslabeled_values.sum()==0:
                return None
            potential_choice=misslabeled_labels[misslabeled_values==torch.max(misslabeled_values)]
            chosen_label=potential_choice[torch.randint(len(potential_choice), (1,))]
        return chosen_label.item()


def global_rule_selection(gt,seg,ignore_label=None):
    """3d rule based on the error size to choose the pixel"""
    errors=(gt!=seg).float()
    if ignore_label is not None :
        ignore_mask=gt!=ignore_label
        errors=errors*ignore_mask 
    else:
        ignore_mask=torch.zeros(1,device=torch.device('cuda:0'))
    proba_click=get_probabilities(errors,ignore_mask)
    proba_with_treshold=torch.where(proba_click>2,proba_click,0)
    if proba_with_treshold.sum()==0:
        return None,None
    try:
        random_pick = torch.multinomial(proba_with_treshold,1,replacement=True) #normalisation is done by the torch function
    except:
        breakpoint()             
    click=torch.nonzero(errors)[random_pick].int().squeeze(0)
    chosen_label=gt[click[0],click[1],click[2]].int().item()

    return click,chosen_label


def select_pixel_3d(gt,seg,mode='global',ignore_label=None):
    """function that given a prediction and its associate groundtruth choose a pixel for guidance with respect to a specified rule (max, proportionnal or global)
    gt and seg should have the same dimension (*,d,h,w)"""
    if mode == 'global':
        # return global_rule_selection(gt,seg,ignore_label)
        # return scipy_get_one_click(gt,seg,ignore_label)
        return get_one_click_morpho(gt,seg,ignore_label)
    
    else:
        chosen_label=choose_label(gt,seg,click_rule=mode)
        if chosen_label==None:
                            print('no error on this image ')
                            return None,None
        error_mask=gt!=seg
        label_mask=gt==chosen_label #ignore label mask 
        errors=error_mask*label_mask
        proba_click=get_probabilities(errors.float())
        proba_with_treshold=torch.where(proba_click>2,proba_click,0)
        if proba_with_treshold.sum()==0:
            return None,None
        try:
            random_pick = torch.multinomial(proba_with_treshold,1,replacement=True) #normalisation is done by the torch function
        except:
            breakpoint()             
        click=torch.nonzero(errors)[random_pick].int().squeeze(0)
        chosen_label=gt[click[0],click[1],click[2]].int().item()
        
        return click,chosen_label
    
def click_simulation_test(self,data,target,training_mode=True,click_mode='global'):
    """ main function that allows us to simulate clicks
    training_mode : True if you use this function during training, False if it's during validation"""
    b,c,d,h,w= data.size()  # batch, channel, depth, height, width
    groundtruth=target[0]
    click_mask= torch.full((b, c - 1,d, h, w), 0, device=self.device, dtype=torch.float)
    if np.random.binomial(n=1,p=self.nbr_supervised): #choosing if the batch is gonna be train with clicks or not   
        self.network.eval() #putting the model in inference mode, needed to simulate click 
        for k in range(self.max_iter):
        #we first want to get map probabilities
            if do_simulate(k,self.max_iter,training_mode):
                # using current network to have prediction & probabilities 
                with torch.no_grad():
                    data[:,1:]=apply_gaussian_bluring(click_mask,(1,2,2),5)
                    logits = self.network(data)
                    # probabilities = torch.softmax(logits[0],dim=1)
                    # prediction = torch.max(probabilities,dim=1)[1]
                    prediction = torch.max(logits[0],dim=1)[1]
                for nimage in range(b):
                        start=time.time()
                        click, chosen_label=select_pixel_3d(groundtruth[nimage,0],prediction[nimage],mode=click_mode,ignore_label=self.label_manager.ignore_label)
                        stop=time.time()
                        print(f'checkpoint 4 : {stop-start} s')
                        # breakpoint()
                        if click==None:
                            print('no error big enough,skiping image{} at step {}'.format(nimage,k))
                            continue

                        # add click to click map
                        click_mask[nimage,chosen_label,click[0],click[1],click[2]] = 1                                                     
            else:
                break
        # here we smoothed the click data
        data[:,1:]=apply_gaussian_bluring(click_mask,2,5)
        if training_mode:
            self.network.train() #putting the model back to training mode 
            print('all click generated!, starting gradient descent...')
    return data,data[:,1:]


def label_weight(seg,nlabel):
    """ funtion to compute proportion of pixel that belongs to each labels """
    count=[]
    for k in range(nlabel):
        count.append((seg==k).sum())
    count=torch.tensor(count,device='cuda:0')
    return count.float()/count.sum()

class New_loss(RobustCrossEntropyLoss):
    def __init__(self):
        super().__init__()

    def forward(self,net_output,gt):
        pred=torch.max(net_output,dim=1)[1]
        # nlabel=torch.max(pred.unique().shape[0],gt.unique().shape[0])
        regu=(torch.abs(label_weight(gt,9)-label_weight(pred,9))).sum()
        l=super().forward(net_output.float(),gt.float())
        
        return l+regu

def windowed_tensor(T,center,radius):
    """function to windowed a tensor center should be a 3D point"""
    B,C,D,H,W=T.shape
    d,h,w=center

    start_d=max(0,d-radius)
    end_d=min(d+radius,D)
    start_h=max(0,h-radius)
    end_h=min(h+radius,H)
    start_w=max(0,w-radius)
    end_w=min(w+radius,W)

    return T[:,:,start_d:end_d, start_h:end_h, start_w:end_w]
    
def local_tensor(T,pt,radius):
    """function to get a localize part of a tensor"""
    if len(T.shape)==5:
        z,x,y=pt
        return T[:,:,z-radius:z+radius,x-radius:x+radius,y-radius:y+radius]
    elif len(T.shape)==4:
        x,y=pt
        return T[:,:,x-radius:x+radius,y-radius:y+radius]

class click_penalty_loss(New_loss):

    def __init__(self):
        super().__init__()
        self.click_map=None
        self.radius=5
        self.alpha=1
        self.saver=0
    def forward(self,net_output,gt):
        if self.click_map==None:
            return super().forward(net_output,gt)
        click_list=torch.nonzero(self.click_map)
                                 
        if len(click_list)==0:
            return super().forward(net_output,gt)
        
        click_loss=0
        for click in click_list:
            pt=click[2:]
            click_loss+=super(New_loss,self).forward(windowed_tensor(net_output,pt,self.radius),windowed_tensor(gt,pt,self.radius))
        if torch.isnan(click_loss): click_loss=self.saver
        else:
            self.saver=click_loss
        print('alpha:',self.alpha,'click loss:',click_loss.item(),'real_loss:',(click_loss/len(click_list)).item())
        return self.alpha*super().forward(net_output,gt)+((1-self.alpha)/len(click_list))*click_loss
    
class test_loss(DC_and_CE_loss):
    """ click penalty loss with the standard DC+CE loss as basis"""
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                 dice_class)
        self.click_map=None
        self.radius=5
        self.alpha=1
        self.saver=0
        self.ds_iterator=0

    def forward(self,net_output,gt):
        if self.click_map==None:
            return super().forward(net_output,gt)
        click_list=torch.nonzero(self.click_map)
                                 
        if len(click_list)==0:
            return super().forward(net_output,gt)
        
        if self.ds_iterator%4!=0:
            self.ds_iterator=(self.ds_iterator+1)%4
            return super().forward(net_output,gt)
        else:
            self.ds_iterator=(self.ds_iterator+1)%4
        click_loss=0
        for click in click_list:
            pt=click[2:]
            click_loss+=super().forward(windowed_tensor(net_output,pt,self.radius),windowed_tensor(gt,pt,self.radius))
        if torch.isnan(click_loss): click_loss=self.saver
        else:
            self.saver=click_loss
        print('alpha:',self.alpha,'click loss:',click_loss.item(),'real_loss:',(click_loss/len(click_list)).item())
        return self.alpha*super().forward(net_output,gt)+((1-self.alpha)/len(click_list))*click_loss

class test_loss2(DC_and_CE_loss):
    """ click penalty loss with the standard DC+CE loss as basis,
    click map is a heat map with gaussian bluring """
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                    dice_class=SoftDiceLoss):
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                    dice_class)
        self.click_map=None
        self.alpha=1

    def forward(self,net_output,gt):
        #variable to build : click_map, radius, alpha)
        if self.click_map==None:
            return super().forward(net_output,gt)
        if self.click_map.sum()==0:
            return super().forward(net_output,gt)
        gt_masked=torch.where(self.click_map.unsqueeze(1)==1,gt,self.ignore_label)
        click_loss=super().forward(net_output,gt_masked)
        print('alpha:',self.alpha,'click_loss:',click_loss.item())
        return self.alpha*super().forward(net_output,gt)+((1-self.alpha))*click_loss

class loss_P0_and_click_region(DC_and_CE_loss):
    """ click penalty loss with the standard DC+CE loss as basis,
    click map is a heat map with gaussian bluring """
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                    dice_class=SoftDiceLoss):
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                    dice_class)
        self.click_map=None
        self.alpha=1
        self.net_output0=None
        
    def forward(self,net_output,gt):
        #variable to build : click_map, radius, alpha)
        if self.click_map==None:
            return super().forward(net_output,gt)
        if self.click_map.sum()==0:
            return super().forward(net_output,gt)
        if self.ignore_label is not None:
            is_ignored = torch.any(gt.squeeze() == self.ignore_label,axis=[2,3])
            is_seg = 1 - (is_ignored.int().unsqueeze(2).unsqueeze(2))
            self.click_map = self.click_map * is_seg
        gt_masked = torch.where(self.click_map.unsqueeze(1) == 1,gt,self.ignore_label)         
        click_loss = super().forward(net_output,gt_masked)
        print('alpha:',self.alpha,'click_loss:',click_loss.item())
        return self.alpha*super().forward(self.net_output0,gt) + ((1-self.alpha)) * click_loss
    
class DeepSupervisionWrapper(torch.nn.Module):
    """ wrapper to make a the deep supervision work with click penalty """
    def __init__(self, loss,scales, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        self.click_map=None
        self.net_output=[None]*len(weight_factors)
        self.scales=scales

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors
        if self.click_map==None:
            return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        else:
            result=0
            for i, inputs in enumerate(zip(*args)):
                    if weights[i] != 0.0:
                            new_shape=[round(i*j) for i,j in zip(self.click_map.shape[1:],self.scales[i])]
                            self.loss.click_map = (TF.interpolate(self.click_map.float().unsqueeze(1),mode='nearest-exact',size=new_shape)>0.5).float().squeeze()
                            self.loss.net_output0 = self.net_output0[i]
                            result+=weights[i]*self.loss(*inputs)

            return result
        
if __name__=='__main__':
    A=click_penalty_loss()
    click_mask=torch.randint(0,2,(2,9,5,5,3))
    B=torch.randint(0,10,(5,5,3))
    C=[torch.randint(0,10,(5,10,5,3))]*5
    D=A.forward(C,B,click_mask,2,0.2)
    print(D)