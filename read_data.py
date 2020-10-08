import cv2
import os
import numpy as np
import random

class read_data:
    def __init__(self, args):
        self.folder = args.data_set_dir
        self.dim_x = args.data_input_dim
        self.dim_y = args.data_output_dim
        self.data_size = args.data_size
        self.train_size = args.train_size
        self.val_size = args.val_size
        self.test_size = args.test_size
        self.image_set = args.image_set
        
        
    def read_files(self):
        # Splitting the data into validation, test and train
        val_test_list = random.sample(range(self.data_size), self.test_size+ self.val_size)

        train_list = []
        val_list = []
        test_list = random.sample(val_test_list, self.test_size)
        index_v = 0
        index_tr = 0
        index_te = 0

        for i in range(self.data_size):
            if (i in val_test_list):
                pass
            else:
                train_list.append(i)
        
        for i in val_test_list:
            if (i in test_list):
                pass
            else:
                val_list.append(i)

        # Reads all the files in the specified directory
        for r,d,f in os.walk(self.image_set):
            pass
        # Sorts them to get the image and it's corresponding mask next to each other
        f.sort()
        for i in range(self.data_size):
            img = cv2.imread(self.image_set + f[2*i])
            out = cv2.imread(self.image_set + f[(2*i)+1], 0)
            ret,thresh = cv2.threshold(out,127,255,cv2.THRESH_BINARY) # would work for this dataset check for others
            thresh = np.where(thresh==255, 1, 0)
            self.y = np.stack((thresh, thresh, thresh), 2) # class 0-background and 1-crop
            
            # Saves the images and masks in their corresponding directories
            if i in val_list:
                cv2.imwrite(self.folder + "val_img/"+ str(index_v)+ ".jpg", np.array(img, dtype = np.uint8))
                cv2.imwrite(self.folder + "val_mask/"+ str(index_v)+ ".jpg", np.array(self.y, dtype = np.uint8))    
                index_v += 1
            if i in train_list:
                cv2.imwrite(self.folder + "train_img/"+ str(index_tr)+ ".jpg", np.array(img, dtype = np.uint8))
                cv2.imwrite(self.folder + "train_mask/"+ str(index_tr)+ ".jpg", np.array(self.y, dtype = np.uint8))    
                index_tr += 1
            if i in test_list:
                cv2.imwrite(self.folder + "test_img/"+ str(index_te)+ ".jpg", np.array(img, dtype = np.uint8))
                cv2.imwrite(self.folder + "test_mask/"+ str(index_te)+ ".jpg", np.array(self.y, dtype = np.uint8))
                index_te += 1                
        return
