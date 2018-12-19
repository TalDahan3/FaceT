import keras
from keras.layers import Conv2D, Dense, Layer
import keras.backend as K
import tensorflow as tf
import numpy as np

class RoiLayer (Layer):
    """ Algorithm:
        https://deepsense.ai/region-of-interest-pooling-explained/
    """
    def __init__ (self,num_of_rois,output_dim, **kwargs):
        self.output_dim = output_dim
        self.num_of_rois = num_of_rois
        super(RoiLayer,self).__init__(**kwargs)

    def build (self, input_shape):
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[1],self.output_dim), initializer = 'zero',trainable = True)
        self.nb_channels = input_shape[0][3]
        super(RoiLayer,self).build(input_shape)

    def call (self,in_layers):
        assert (len(in_layers) == 2) #verifying to inputs to layer
        featMap = in_layers[0] #featrue map input
        rois = in_layers[1] # proposed roi
        xAxis = 1
        yAxis = 2

        for roi_idx in range(self.num_of_rois):
            
            maxRegions = list()

            ### TODO: verify this data structure is what we get from RPN ###
            x1 = rois[0, roi_idx, 0] #upper left
            y1 = rois[0, roi_idx, 1] 
            x2 = rois[0, roi_idx, 2] #lower right
            y2 = rois[0, roi_idx, 3] 

            #calculating num of windows
            assert ((x2 > x1) and (y2 > y1))
            totalWidth = featMap[xAxis]*(x2-x1)
            totalHigth = featMap[yAxis]*(y2-y1)
            
            num_of_cropsX = np.floor(totalWidth / float(self.output_dim))
            xs = list()
            for i in range(num_of_cropsX-1):
                xs.append(self.output_dim)
            xs.append(self.output_dim + (totalWidth % self.output_dim))

            
            num_of_cropsY = np.floor(totalWidth / float(self.output_dim))
            ys = list()
            for i in range(num_of_cropsY-1):
                ys.append(self.output_dim)
            ys.append(self.output_dim + (totalHigth % self.output_dim))

            #moving sliding window
            #croping image
            xStart = x1
            
            for i in range(xs.__len__):
                xStart = xStart + self.output_dim
                yStart = y1
                for j in range(ys.__len__):
                    yStart = yStart + self.output_dim
                    tfx1 = tf.cast(xStart*featMap[xAxis],dtype = tf.int32)
                    tfwidth = tf.cast(xs[i],dtype = tf.int32)
                    tfy1 = tf.cast(yStart*featMap[yAxis],dtype = tf.int32)
                    tfhighth = tf.cast(ys[j],dtype = tf.int32)
                    croppedImg = tf.image.crop_to_bounding_box(featMap,tfy1,tfx1,tfhighth,tfwidth)
                    ### for each region perform max pooling ###
                    maxRegions += K.pool2d(croppedImg,1,1,pool_mode='max')
            
            final_output = K.concatenate(maxRegions, axis=0)
            final_output = K.reshape(final_output, (1, self.num_of_rois, self.output_dim, self.output_dim, self.nb_channels))

            return final_output
            #cut to smaller regions
                
            ### concatanate all maxed regions

    def compute_output_shape(self, input_shape):        
        return None, self.num_of_rois, self.output_dim, self.output_dim, self.nb_channels

    