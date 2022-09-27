import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from models.model import model
import pytorch_lightning as pl


parser = ArgumentParser()

# generic arguments
parser.add_argument('--results_folder', type = str, default =  './results')
args = parser.parse_args()


device = 'cuda:0'  


class Net(pl.LightningModule):
    """
    Event-based Deblurring Net
    """
    def __init__(self, hparams):
        super(Net, self).__init__()
        

  
        self.deblur_net = model(8,5) 
        self.deblur_net.to(device)

        
 

    def forward(self,  u,e, e_unsq):
        """
        :param u: input blurry image
        """

        # forward: deblur net
        x = torch.cat((u, e_unsq), 1)
        y = self.deblur_net(x,e)
        return y



# dataloader


# pretrained model
checkpoint_folder = './checkpoints/'
print('Starting model')
experiment = 'Original'
file = 'model.ckpt' 

    
    
PATH_TO_CHECKPOINT = os.path.join(checkpoint_folder,file)
 

print('loading checkpoints')
net = Net.load_from_checkpoint(PATH_TO_CHECKPOINT, strict=False)
net.freeze()
net.to(device)





from dataloader.dataloader import GoProDataset as GoProDataset_Multiple
dataset = GoProDataset_Multiple(f'./examples/',
                        full_image = True)





test_folder = experiment 
if not os.path.exists(os.path.join(args.results_folder,test_folder)):
    os.makedirs(os.path.join(args.results_folder, test_folder), exist_ok=True)  

for idx in range(len(dataset)):
    sample = dataset.__getitem__(idx)
    vg,b_gt,vg_un_0 = sample['voxel_grid'],sample['blur'],sample['voxel_grid_un_0']


    b_gt = b_gt.to(device=device, dtype=torch.float32)

    vg = vg.unsqueeze(0)    
    vg = vg.to(device=device, dtype=torch.float32)
    

    vg_un_0 = vg_un_0.unsqueeze(0)    
    vg_un_0 = vg_un_0.to(device=device, dtype=torch.float32)

    outputs=net(b_gt,vg,vg_un_0)
    output = b_gt + outputs[0]
    pred = torch.clamp(output, -1.0, 1.0)

    save_image(pred[0]/2.0+0.5,os.path.join(args.results_folder, test_folder)+'/'+sample['file_name']+"_"+"_pred.png")
    print('image saved in ', os.path.join(args.results_folder, test_folder)+'/'+sample['file_name']+"_"+"_pred.png")

