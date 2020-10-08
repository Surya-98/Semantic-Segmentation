import cv2
import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array

# Function to create output labels
def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            if(labels[i][j]< n_labels):
                x[i, j, labels[i][j]] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x

# Dataset Generator
def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            img_path = img_dir + str(lists[i]) + ".jpg"
            # original_img = cv2.imread(img_path, 0)  # for grayscale image as input
            original_img = cv2.imread(img_path)
            resized_img = cv2.resize(original_img, (dims[0], dims[1]))       # Resizing the image to the network input size
            array_img = img_to_array(resized_img) / 255                      # Making the input between 0 and 1
            imgs.append(array_img)

            # masks
            original_mask = cv2.imread(mask_dir + str(lists[i]) + ".jpg")
            resized_mask = cv2.resize(original_mask, (dims[0], dims[1]), interpolation = cv2.INTER_NEAREST) # Resizing the image to the network output size while making sure the classes are maintained
            array_mask = category_label(resized_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels
        

