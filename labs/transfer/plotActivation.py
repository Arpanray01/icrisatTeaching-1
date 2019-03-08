'''
File: plotActivation.py
Project: transfer
File Created: Thursday, 7th March 2019 9:19:29 am
Author: Romain GAUTRON (r.gautron@cgiar.org)
-----
Last Modified: Thursday, 7th March 2019 9:19:40 am
Modified By: Romain GAUTRON (r.gautron@cgiar.org>)
-----
Copyright 2019 Romain GAUTRON, CIAT
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import preprocess_input
K.clear_session()
K.set_learning_phase(0)

def plotActivationMap(fullModel,testImagePath,img_width,img_height,savingName='leafActivation.jpg'):
    """[summary]
    Fuctions that shows from an image the activated region in and Xception based model.
    Will save a new image with highlighted activation zones.
    Arguments:
        fullModel {[type] keras model} -- [description] Xception based model
        testImagePath {[type] string} -- [description] path of the image we want to see the activation
        img_width {[type] int} -- [description] width of the image
        img_height {[type] int} -- [description] width of the image
    
    Keyword Arguments:
        savingName {str} -- [description] (default: {'leafActivation.jpg'}) path where to solve the resulting image with highlighted activation zones
    """
    testImage = load_img(testImagePath,target_size=(img_width, img_height))
    testImage = img_to_array(testImage)
    testImage = np.expand_dims(testImage,axis=0)
    testImage = preprocess_input(testImage)

    pred = fullModel.predict(testImage)
    print('Predicted class: {}'.format(pred))


    last_conv_layer = fullModel.get_layer('block14_sepconv2')
    grads = K.gradients(fullModel.output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([fullModel.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([testImage])
    for i in range(2048):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    testImageCv2 = cv2.imread(testImagePath)
    heatmap = cv2.resize(heatmap, (testImageCv2.shape[1], testImageCv2.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposedImages = heatmap * 0.4 + testImageCv2
    cv2.imwrite(savingName, superimposedImages)