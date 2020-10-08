import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
from time import time
from read_data import read_data
import argparse
from generator import data_gen_small
from model import segnet
import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from datetime import date
import tensorflow as tf
from keras import backend as K
import sys
import argparse
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

# Class that holds the values of all the parameters
class params():
    def __init__(self, model_no, model_name):
        self.primary_dir = "/home/killswitch/Desktop/Vision_Robotics/"               # The main directory's path
        self.image_set = self.primary_dir + "images/cultivator-170815-160439/"       # The sub path to the images with mask
        self.data_set_dir = self.primary_dir + "codes/dataset/"                      # The sub path to store and retrieve the dataset
        self.save_dir = self.primary_dir + "codes/output/"                           # The sub path to store the output
        self.trainimg_dir = self.data_set_dir + "train_img/"                              
        self.trainmsk_dir = self.data_set_dir + "train_mask/"
        self.valimg_dir = self.data_set_dir + "val_img/"
        self.valmsk_dir = self.data_set_dir + "val_mask/"
        self.testimg_dir = self.data_set_dir + "test_img/"
        self.testmsk_dir = self.data_set_dir + "test_mask/"
        self.data_size = 524                                                         # The total number of images
        self.val_size = int(self.data_size*0.2)                                      # 20 percentage of data for validation
        self.test_size = int(self.data_size*0.2)                                     # 20 percentage of data for testing
        self.train_size = self.data_size - self.val_size - self.test_size            # 60 percentage of data for training
        self.data_input_dim = [self.data_size, 480, 752, 3]                          # Image dimensions
        self.data_output_dim = [self.data_size, 480, 752, 3]                         # Mask dimensions
        self.batch_size = 10                                                         # Batch Size for training
        self.n_epochs = 10                                                           # Number of epochs
        self.epoch_steps = 100                                                        # Steps per epoch
        self.val_steps = 10                                                          # Validation steps
        # self.test_steps = 10 
        self.n_labels = 2
        self.input_shape = (256, 256, 3)                                             # Network Input Size
        self.kernel = 3                                                              # Kernel Size
        self.pool_size = (2, 2)                                                      # Pool Size
        self.output_mode = "softmax"                                                 # Output Mode        
        self.loss = "categorical_crossentropy"                                       # Loss Type
        self.optimizer = "adam" #"adadelta"                                                  # Optimizer Type
        today = date.today()
        self.weight = model_name
        d = today.strftime("%b-%d-%Y")
        if(model_no!= None):
            self.model_number = int(model_no)
            self.new_model_name = d + "_" + str(self.n_epochs) + "_" + str(self.model_number) 

# Function to prepare the data by converting to the required format before training
def prep_data(args):
    read = read_data(args)
    read.read_files()
    # read.save()

# Function to freeze the variables in tensorflow graph
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# Function to train the network
def train(args):
    train_list = range(args.train_size)
    val_list = range(args.val_size)

    # set the necessary directories
    trainimg_dir = args.trainimg_dir
    trainmsk_dir = args.trainmsk_dir
    valimg_dir = args.valimg_dir
    valmsk_dir = args.valmsk_dir

    train_gen = data_gen_small(
        trainimg_dir,
        trainmsk_dir,
        train_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels
    )
    val_gen = data_gen_small(
        valimg_dir,
        valmsk_dir,
        val_list,
        args.batch_size,
        [args.input_shape[0], args.input_shape[1]],
        args.n_labels
    )

    model = segnet(
        args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode, args.model_number
    )
    print(model.summary())

    with open(args.save_dir + 'model'+str(args.model_number)+'.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    model.compile(loss=args.loss, optimizer=args.optimizer, metrics=["accuracy"])

    print("Model Compiled")
    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    model.fit_generator(
        train_gen,
        steps_per_epoch=args.epoch_steps,
        epochs=args.n_epochs,
        validation_data=val_gen,
        validation_steps=args.val_steps,
        callbacks =[tensorboard]
    )
    model.save(args.save_dir + args.new_model_name + ".hdf5")
    print("Model Saved")

    K.set_learning_phase(0)
    session = K.get_session()
    init = tf.global_variables_initializer()
    session.run(init)
    frozen_graph = freeze_session(session,
                              output_names=[out.op.name for out in model.outputs])
    print([out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, args.save_dir, args.new_model_name +".pb", as_text=False)
    session.close()
    file1 = open(args.save_dir + "recent.txt", "w+")
    file1.write(args.new_model_name)
    file1.close()


# Function to load the specified model and save the output test images
def visualize(args):
    testimg_dir = args.testimg_dir
    testmsk_dir = args.testmsk_dir
    test_list = range(args.test_size)

    file1 = open(args.save_dir + "recent.txt", "r")
    args.weight = file1.read()
    file1.close()
        

    try:
        model = segnet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode, int(args.weight[-1]))
        model.load_weights(args.save_dir+args.weight+".hdf5")
        print(args.save_dir+args.weight+".hdf5")
    except:
        print('The model name entered does not exist')
    else:
        for i in test_list:
            img = []
            
            # If grayscale image is used as input
            # original_img = cv2.imread(testimg_dir+str(i)+".jpg", 0)
            # resized_img = cv2.resize(original_img, (args.input_shape[0], args.input_shape[1]))
            # imag = np.stack([resized_img, resized_img, resized_img], axis=2)
            # array_img = img_to_array(resized_img) / 255
            # img.append(imag)

            original_img = cv2.imread(testimg_dir+str(i)+".jpg")
            resized_img = cv2.resize(original_img, (args.input_shape[0], args.input_shape[1]))
            array_img = img_to_array(resized_img) / 255
            img.append(resized_img)
            
            array_img = np.expand_dims(array_img, axis=0)
            output = model.predict(array_img)
            mask = cv2.imread(testmsk_dir+str(i)+".jpg")
            resized_mask = cv2.resize(mask*255, (args.input_shape[0], args.input_shape[1]), interpolation = cv2.INTER_NEAREST)
            img.append(resized_mask)
            resized_image = np.reshape(output[0]*255,(args.input_shape[0], args.input_shape[1], args.n_labels))
            img.append(resized_image[:,:,1])
            fig = plt.figure()
            ax = []
            for j in range(3):
                ax.append(fig.add_subplot(1, 3, j+1))
                if j == 0:
                    plt.imshow(img[j])
                else:
                    plt.imshow(img[j], cmap='gray', vmin=0, vmax=255)

            ax[0].title.set_text("Original Image")
            ax[1].title.set_text("Ground Truth")
            ax[2].title.set_text("Predicted Mask")
            
            fig.savefig(args.save_dir+"images/"+ str(i)+ ".png", dpi=fig.dpi)
            plt.close('all')

# Main function
def main():
    parser = argparse.ArgumentParser(description='Image Segmentation using Deep Learning')
    parser.add_argument('-p', '--prep_data', help='Prepare the data and split for training, validation and testing', action='store_true')
    parser.add_argument('-t' , '--train', help='Train and save the network, enter the model number as argument')
    parser.add_argument('-v', '--visualize', help='Visualize the output of the specified network, enter the model name as argument')

    arguments = parser.parse_args()
    args = params(arguments.train, arguments.visualize)
    
    if arguments.prep_data:
        prep_data(args)
    if arguments.train!= None:
        train(args)
    elif arguments.visualize!= None:
        visualize(args)

if __name__=="__main__" :
    main()
