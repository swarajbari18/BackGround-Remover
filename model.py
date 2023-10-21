import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import (Input,
                                    Conv2D,
                                    BatchNormalization, 
                                    Activation,
                                    MaxPool2D,
                                    UpSampling2D,
                                    Concatenate,
                                    Add
                                )

def convolution_block ( input, output_channel, dilation_rate ):

    x = Conv2D ( output_channel, 3, padding= "same", dilation_rate= 1) ( inputs )
    x = BatchNormalization () ( x )
    x = Activation ( 'relu' ) ( x )

    return x

def Residual_U_block ( inputs, output_channel, intermidiate_chaneels, num_layers, dilated_convolution_rate= 2):
    ''' Initial Convolution Layer '''

    x = convolution_block ( input= inputs, output_channel= output_channel )
    initial_features = x

    ''' Encoder '''

    skip = [ ]
    
    x = convolution_block ( x, intermidiate_chaneels )
    skip.append( x )

    for i in range ( num_layers - 2 ):
        x = MaxPool2D ( size= (2, 2 ) ) ( x )     # Downsampling
        x = convolution_block ( x, intermidiate_chaneels )
        skip.append ( x )

    ''' Bridge ''' # In the paper it is mentioned as Encoder only, but let me call it a bridge

    x = convolution_block ( x, intermidiate_chaneels, dilation_rate= dilated_convolution_rate )

    ''' Decoder '''

    skip.reverse ( )   # try adding deque here

    x = Concatenate ( ) ( [x, skip[ 0 ] ] )
    x = convolution_block ( x, intermidiate_chaneels )

    for i in range ( num_layers - 3 ):

        x = UpSampling2D ( size= ( 2, 2 ), interpolation= 'bilinear' ) ( x )
        x = Concatenate ( ) ( [ x, skip [ i + 1 ] ]) 
        x = convolution_block ( x, intermidiate_chaneels )

    ''' Last Decoder Layer '''

    x = UpSampling2D ( size= ( 2, 2 ), interpolation= 'bilinear' ) ( x )
    x = Concatenate ( ) ( [ x, skip [ -1 ] ]) 
    x = convolution_block ( x, output_channel )


    ''' Add '''

    x = Add ( x, initial_features )

    return x



def Residual_U_block_in_Encoder5_6_and_Decoder_5 ( input, output_chaneel, intermidiate_channel ):   

    ''' Initial convolution '''

    x0 = convolution_block ( input, output_chaneel, dilation_rate= 1 )


    ''' Encoder '''
    x1 = convolution_block ( x0, intermidiate_channel, dilation_rate= 1 )
    x2 = convolution_block ( x1, intermidiate_channel, dilation_rate= 2 )
    x3 = convolution_block ( x2, intermidiate_channel, dilation_rate= 4 )

    ''' Bridge '''

    x4 = convolution_block ( x3, intermidiate_channel, dilation_rate= 8 )


    ''' Decoder '''

    x = Concatenate ( ) ( [ x4, x3 ] )
    x = convolution_block ( x, intermidiate_channel, dilation_rate= 4 )

    x = Concatenate ( ) ( [ x, x2 ] )
    x = convolution_block ( x, intermidiate_channel, dilation_rate= 2 )

    x = Concatenate ( ) ( [ x, x1 ] )
    x = convolution_block ( x, output_chaneel, dilation_rate= 1 )

    ''' Addition '''

    x = Add ( ) ( [ x, x0 ] )


    return x





if __name__ == '__main__' :

    pass