import kornia
import torch
from tools import event, representation, io_tools
from tools.event import  EventSequence
import numpy as np
import cv2
import os



def _prepare_to_collate(np_ndarray, to_0_1=False):
    # HxWxC -> CxHxW
    np_ndarray = np_ndarray.transpose(2, 0, 1).copy()
        
    # Normalize to [0, 1] for images
    if to_0_1:
        np_ndarray = np_ndarray.astype(np.float32) / np.iinfo(np_ndarray.dtype).max
            
    return torch.from_numpy(np_ndarray).float()

class GoProDataset:
    def __init__(self,samples_path, 
                 crop_size = 448,      
                number_of_time_bins = 5,
                full_image = False):
        self.crop_size = crop_size
        self.number_of_time_bins =number_of_time_bins                 
        self.full_image = full_image
        self.samples_path = samples_path
        self.blur_folder = os.path.join(samples_path, 'blur') 
        self.num_frames = 11
        self.files = []
        try:
            self.files = io_tools.find_files_by_template(self.blur_folder, '.png')
        except FileNotFoundError:
            self.files = []
            print('samples not found')
        print('num images in ', self.blur_folder,' is equal to ',len(self.files))


    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, example_index):
        sample = self.files[example_index]
        file_name = sample.split('/')[-1]
        file_name = file_name.split('.')[0]
        
        event_folder = os.path.join(self.samples_path, 'events', file_name+'.npy') 
        sharp_folder = os.path.join(self.samples_path, 'sharp', file_name+'.png') 


        
        blur = cv2.imread(sample)
        blur_tensor = kornia.image_to_tensor(blur[:, :, ::-1].copy(), keepdim = False).float()/255.0
        
        try:
            sharp = cv2.imread(sharp_folder)
            sharp_tensor = kornia.image_to_tensor(sharp[:, :, ::-1].copy(), keepdim = False).float()/255.0
            sample_dictionary['latent_images'] =sharp_tensor*2.0-1.0
        except:
            print(' not gt found for image ', file_name)
        events = np.float32(np.load(event_folder))
        timestamps = np.arange(0.0, 1.0 + 1e-7, 1 / (self.num_frames-1))
        event_sequence = event.EventSequence(events, blur.shape[0],blur.shape[1], 0.0, 1.0)

        voxel_grid = []

        for i in range(len(timestamps)-1):
            voxel_grid_i = representation.to_trilinear_voxel_grid(
                event_sequence.filter_by_timestamp(timestamps[i], timestamps[i+1]), 
                self.number_of_time_bins, 
                dtype=torch.float32
            ).numpy().transpose(1, 2, 0)
            voxel_grid.append(voxel_grid_i)
            voxel_grid_no_chuncks = []
            voxel_grid_i = representation.to_trilinear_voxel_grid(
                event_sequence.filter_by_timestamp(0.0, 1.0), 
                self.number_of_time_bins, 
                dtype=torch.float32
            ).numpy().transpose(1, 2, 0)
            voxel_grid_no_chuncks.append(voxel_grid_i)


        # prepare sample dictionary
        sample_dictionary = {}
        sample_dictionary['blur'] =  blur_tensor*2.0-1.0
        vg = torch.zeros(13,self.number_of_time_bins,blur.shape[0],blur.shape[1])
        for i in range(len(voxel_grid)):
            vg[i,:,:,:]= _prepare_to_collate(voxel_grid[i], to_0_1 = False)
        sample_dictionary['voxel_grid'] = vg
        sample_dictionary['file_name'] =file_name
        vg_no_c= _prepare_to_collate(voxel_grid_no_chuncks[0], to_0_1 = False)    
        sample_dictionary['voxel_grid_un_0'] = vg_no_c


        return sample_dictionary
