import keras
import numpy as np
from PIL import Image

class DataGenerator (keras.utils.Sequence):
    def __init__(self, list_images, list_annonations, batch_size=32, dim=(224,224,3),n_channels=1):
        #Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_annonations = list_annonations
        self.list_images = list_images
        self.n_channels = n_channels
        self.on_epoch_end()

    def __data_generation(self, list_images_temp, list_annonations_temp):
        ###########################################################################
        # i = 0
        # for data in img_class:        
        #     rectangles = []
        #     for anno in data['annotation']:
        #         #image original size
        #         width = anno['imageWidth']
        #         hight = anno['imageHeight']
        #         #calculating ratio
        #         rHight = tragetHight / hight
        #         rWidth = targetWidth / width
        #         #get 4 points of annotation
        #         x1 = anno['points'][0]['x']
        #         y1 = anno['points'][0]['y']
        #         x2 = anno['points'][1]['x']
        #         y2 = anno['points'][1]['y']
        #         #create rectangle
        #         rectangles.append(plt.Rectangle((width*rWidth*x1,hight*rHight*y1),(x2-x1)*width*rWidth,(y2-y1)*hight*rHight,fill = False, color = 'green'))
        #     img = Image.open(r"Dataset/Images/Train/"+str(i)+".jpg")
        #     img = img.resize((tragetHight,targetWidth))
        #     fig,axes = plt.subplots(1)
        #     plt.imshow(img)       
        #     #adding rectrangles
        #     for rec in rectangles:
        #         axes.add_patch(rec)
        #     plt.show(img)
        #     i=i+1
        ####################################################################################

        #x - saves the data
        x = np.empty((self.batch_size,*self.dim,self.n_channels))

        #y - saves the annonations (and points)
        points = [{'x','y'}]
        y = np.empty((self.batch_size,points))
        # load data
        for i, ID in enumerate(list_images_temp):
            img = Image.open(r"Dataset/Images/Train/"+str(i)+".jpg")
            img = img.resize((self.dim[0],self.dim[1]))
            #Store sample
            x[i,] = img
            #Store points
            for anno in self.list_annonations_temp[i]:
                y[i]['points'][0]['x'] = anno['points'][0]['x']
                y[i]['points'][0]['y'] = anno['points'][0]['y']
                y[i]['points'][1]['x'] = anno['points'][1]['x']
                y[i]['points'][1]['x'] = anno['points'][1]['y']

        return x,y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y
# def on_epoch_end(self):
#   'Updates indexes after each epoch'
#   self.indexes = np.arange(len(self.list_IDs))
#   if self.shuffle == True:
#       np.random.shuffle(self.indexes)
