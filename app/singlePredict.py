import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

#size of image
my_model = "../trained-model/libras_model_20-05-19.h5"
width = 100
height = 100

def predict(img):
  
  from keras.models import load_model
  model = load_model('../trained-model/libras_model_20-05-19.h5')
  
  npImg = np.array(img)

  print('img.shape: ', npImg.shape)

  #Resizes the array to 4 dimensions
  imgToPredict = npImg.reshape(1,width,height,1)
  
  print('imgToPredict.shape: ', imgToPredict.shape)
  print(imgToPredict)
  #Save the array to a file
  #np.save('imgMatriz', lista)

  y_pred = model.predict(imgToPredict)
  print('y_pred: ', y_pred)
  return y_pred
  