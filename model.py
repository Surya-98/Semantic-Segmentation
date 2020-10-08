from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.core import Activation, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import UpSampling2D, MaxPooling2D, Conv2DTranspose

# The layers used here should be supported by tensorrt

def segnet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax", model_number=1):
        
    if(model_number==1):
        # encoder
        inputs = Input(shape=input_shape)

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = Activation("relu")(conv_2)
        conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(conv_2)
        conv_3 = Activation("relu")(conv_3)

        pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_4 = Activation("relu")(conv_4)
        conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_4)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
        conv_6 = Activation("relu")(conv_6)

        pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_6)
        print("Encoder built..")

        # decoder

        # unpool_1 = Conv2DTranspose(128, (kernel, kernel), padding="same", strides=(2,2))(pool_2)
        unpool_1 = UpSampling2D(size=pool_size)(pool_2)
        conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_1)
        conv_7 = Activation("relu")(conv_7)
        conv_8 = Convolution2D(128, (kernel, kernel), padding="same")(conv_7)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(128, (kernel, kernel), padding="same")(conv_8)
        conv_9 = Activation("relu")(conv_9)
        
        unpool_2 = UpSampling2D(size=pool_size)(conv_9)
        # unpool_2 = Conv2DTranspose(64, (kernel, kernel), padding="same", strides=(2,2))(conv_9)
        conv_10 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_2)
        conv_10 = Activation("relu")(conv_10)
        conv_11 = Convolution2D(64, (kernel, kernel), padding="same")(conv_10)
        conv_11 = Activation("relu")(conv_11)

        conv_12 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_11)
        conv_12 = Reshape(
            (input_shape[0] * input_shape[1], n_labels),
            input_shape=(input_shape[0], input_shape[1], n_labels),
        )(conv_12)

        outputs = Activation(output_mode)(conv_12)
        print("Decoder built..")

        model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    elif(model_number==2):
        # encoder
        inputs = Input(shape=input_shape)

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = Activation("relu")(conv_2)
        conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(conv_2)
        conv_3 = Activation("relu")(conv_3)

        pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_4 = Activation("relu")(conv_4)
        conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_4)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
        conv_6 = Activation("relu")(conv_6)

        pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_6)
        conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
        conv_7 = Activation("relu")(conv_7)
        conv_8 = Convolution2D(256, (kernel, kernel), padding="same")(conv_7)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(256, (kernel, kernel), padding="same")(conv_8)
        conv_9 = Activation("relu")(conv_9)

        pool_3 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_9)
        print("Encoder built..")

        # decoder
        unpool_1 = Conv2DTranspose(256, (kernel, kernel), padding="same", strides=(2,2))(pool_3)
        conv_10 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_1)
        conv_10 = Activation("relu")(conv_10)
        conv_11 = Convolution2D(256, (kernel, kernel), padding="same")(conv_10)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(256, (kernel, kernel), padding="same")(conv_11)
        conv_12 = Activation("relu")(conv_12)

        unpool_2 = Conv2DTranspose(128, (kernel, kernel), padding="same", strides=(2,2))(conv_12)
        conv_13 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_2)
        conv_13 = Activation("relu")(conv_13)
        conv_14 = Convolution2D(128, (kernel, kernel), padding="same")(conv_13)
        conv_14 = Activation("relu")(conv_14)
        conv_15 = Convolution2D(128, (kernel, kernel), padding="same")(conv_14)
        conv_15 = Activation("relu")(conv_15)

        unpool_3 = Conv2DTranspose(64, (kernel, kernel), padding="same", strides=(2,2))(conv_15)
        conv_16 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_3)
        conv_16 = Activation("relu")(conv_16)
        conv_17 = Convolution2D(64, (kernel, kernel), padding="same")(conv_16)
        conv_17 = Activation("relu")(conv_17)

        conv_18 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_17)
        conv_18 = Reshape(
            (input_shape[0] * input_shape[1], n_labels),
            input_shape=(input_shape[0], input_shape[1], n_labels),
        )(conv_18)

        outputs = Activation(output_mode)(conv_18)
        print("Decoder built..")

        model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    
    elif(model_number==3):
         # encoder
        inputs = Input(shape=input_shape)

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = Activation("relu")(conv_2)
        conv_3 = Convolution2D(64, (kernel, kernel), padding="same")(conv_2)
        conv_3 = Activation("relu")(conv_3)

        pool_1 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_4 = Activation("relu")(conv_4)
        conv_5 = Convolution2D(128, (kernel, kernel), padding="same")(conv_4)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(128, (kernel, kernel), padding="same")(conv_5)
        conv_6 = Activation("relu")(conv_6)

        pool_2 = MaxPooling2D(pool_size=pool_size, padding="valid")(conv_6)
        print("Encoder built..")

        # decoder

        unpool_1 = Conv2DTranspose(128, (kernel, kernel), padding="same", strides=(2,2), activation = 'relu')(pool_2)
        # unpool_1 = UpSampling2D(size=pool_size)(pool_2)
        conv_7 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_1)
        conv_7 = Activation("relu")(conv_7)
        conv_8 = Convolution2D(128, (kernel, kernel), padding="same")(conv_7)
        conv_8 = Activation("relu")(conv_8)
        # conv_9 = Convolution2D(128, (kernel, kernel), padding="same")(conv_8)
        # conv_9 = Activation("relu")(conv_9)
        
        # unpool_2 = UpSampling2D(size=pool_size)(conv_9)
        unpool_2 = Conv2DTranspose(64, (kernel, kernel), padding="same", strides=(2,2), activation = 'relu')(conv_8)
        conv_10 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_2)
        conv_10 = Activation("relu")(conv_10)
        conv_11 = Convolution2D(64, (kernel, kernel), padding="same")(conv_10)
        conv_11 = Activation("relu")(conv_11)

        conv_12 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_11)
        conv_12 = Reshape(
            (input_shape[0] * input_shape[1], n_labels),
            input_shape=(input_shape[0], input_shape[1], n_labels),
        )(conv_12)

        outputs = Activation(output_mode)(conv_12)
        print("Decoder built..")

        model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    return model
