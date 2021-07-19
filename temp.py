import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

class PennFudanDataset():
    def __init__(self, root, transforms= None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = []
        scandirs = [ os.scandir(teeth_id_input) for teeth_id_input in [ os.path.join(teeth,'input') for teeth in   [teeth_id.path for teeth_id in os.scandir(root)]]]
        for scandir in scandirs:
            for rotation in scandir:
                self.imgs.append(rotation.path)
        self.imgs.sort()
        self.imgs= self.imgs
        #self.types = ['A', ]#'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N'] #'M'

        
        
        #self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        #size = 256,256
        img_path = self.imgs[idx]
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        #img = img.thumbnail(size,Image.ANTIALIAS)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask_blank = np.array([512,512])
        mask1 = Image.open(img_path.replace('input', ( 'A' + os.sep + 'mask' )))
        mask2 = Image.open(img_path.replace('input', ( 'B' + os.sep + 'mask' )))
        mask3 = Image.open(img_path.replace('input', ( 'C' + os.sep + 'mask' )))
        mask4 = Image.open(img_path.replace('input', ( 'D' + os.sep + 'mask' )))
        mask5 = Image.open(img_path.replace('input', ( 'E' + os.sep + 'mask' )))
        mask6 = Image.open(img_path.replace('input', ( 'F' + os.sep + 'mask' )))
        mask7 = Image.open(img_path.replace('input', ( 'G' + os.sep + 'mask' )))
        mask8 = Image.open(img_path.replace('input', ( 'H' + os.sep + 'mask' )))
        mask9 = Image.open(img_path.replace('input', ( 'I' + os.sep + 'mask' )))
        mask10 = Image.open(img_path.replace('input', ( 'J' + os.sep + 'mask' )))
        mask11 = Image.open(img_path.replace('input', ( 'K' + os.sep + 'mask' )))
        mask12 = Image.open(img_path.replace('input', ( 'L' + os.sep + 'mask' )))
        #mask13 = Image.open(img_path.replace('input', ( 'M' + os.sep + 'mask' )))
        mask14 = Image.open(img_path.replace('input', ( 'N' + os.sep + 'mask' )))

        mask1 = np.where(mask1 != 0 ,1,0)
        mask2 = np.where(mask2 != 0 ,1,0)
        mask3 = np.where(mask3 != 0 ,1,0)
        mask4 = np.where(mask4 != 0 ,1,0)
        mask5 = np.where(mask5 != 0 ,1,0)
        mask6 = np.where(mask6 != 0 ,1,0)
        mask7 = np.where(mask7 != 0 ,1,0)
        mask8 = np.where(mask8 != 0 ,1,0)
        mask9 = np.where(mask9 != 0 ,1,0)
        mask10 = np.where(mask10 != 0 ,1,0)
        mask11 = np.where(mask11 != 0 ,1,0)
        mask12 = np.where(mask12 != 0 ,1,0)
        mask14 = np.where(mask14 != 0 ,1,0)
        
        mask_blank[mask1 == 1 ] = 1   
        mask_blank[mask2 == 1 ] = 2  
        mask_blank[mask3 == 1 ] = 3
        mask_blank[mask4 == 1 ] = 4
        mask_blank[mask5 == 1 ] = 5
        mask_blank[mask6 == 1 ] = 6
        mask_blank[mask7 == 1 ] = 7
        mask_blank[mask8 == 1 ] = 8
        mask_blank[mask9 == 1 ] = 9
        mask_blank[mask10 == 1] = 10
        mask_blank[mask11 == 1] = 11
        mask_blank[mask12 == 1] = 12
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Horizontally stacked subplots')
        ax1.imshow(np.asarray(img))
        ax2.imshow(np.asarray(mask_blank))
        
        #for boxes in dataset[data_index][1]['boxes_2']:
        #    xmin, ymin, xmax, ymax = boxes
        #    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
            #ax1.add_patch(rect)
        #    ax2.add_patch(rect)
        print(dataset[data_index][1]['labels'])

a = PennFudanDataset("D:\\Renders\\Render_annotation_ver20\\")
a.__getitem__(0)