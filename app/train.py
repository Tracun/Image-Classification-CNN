import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Size of image
width = 64
height = 64

log_Summary = './Train_Graphs_Logs/summary.txt'
img_model = './Train_Graphs_Logs/model.png'
graph_dir = './Train_Graphs_Logs'

def plotGraph(hist, epochs):
  
  # Plot the training, regression and classification Loss
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, epochs), hist.history["loss"], label="loss")
  plt.savefig("./Train_Graphs_Logs/Loss.png")
  plt.plot(np.arange(0, epochs), hist.history["acc"], label="acc")
  plt.savefig("./Train_Graphs_Logs/Acc.png")
  plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
  plt.savefig("./Train_Graphs_Logs/Val_Loss.png")
  plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
  plt.savefig("./Train_Graphs_Logs/Val_Acc.png")
  plt.title("Training, regression and classification Loss on Dataset")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Training Loss/classification Loss")
  plt.legend(loc="lower left")
  plt.savefig("./Train_Graphs_Logs/training, regression and classification Loss.png")
  
  print(hist.history)
  
def trainModel():

  # CSV that contains images to train
  train = pd.read_csv('../image2csv/output/train_lucaslb.csv')

  train.head()
  labels = train['label'].values
  unique_val = np.array(labels)

  plt.figure(figsize = (18,8))
  sns.countplot(x =labels)
  train.drop('label', axis = 1, inplace = True)

  # Copy the array values
  images = train.values
  images = np.array([i.flatten() for i in images])

  # Transforms labels into an array of classes (24 classes) per 8566 rows. The 0 does not belong to the classes, 1 belongs to the class
  label_binrizer = LabelBinarizer()
  labels = label_binrizer.fit_transform(labels)

  from sklearn.model_selection import train_test_split

  # Parameters to train network
  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
  import keras
  import cv2
  from keras.utils import plot_model
  from keras.models import Sequential
  from keras.callbacks import ModelCheckpoint
  from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
  batch_size = 100
  # Number of classes of network
  num_classes = 3
  # Defines the epochs to train the network
  epochs = 50
  x_train = x_train / 255
  x_test = x_test / 255
  x_train = x_train.reshape(x_train.shape[0], width, height, 1)
  x_test = x_test.reshape(x_test.shape[0], width, height, 1)
  plt.imshow(x_train[0].reshape(width,height))
  model = Sequential()
  
  # Defines the network structure
  model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape=(width, height ,1) ))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
  model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
  model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
  model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
  # Convert 2D array to 1D Array
  model.add(Flatten())
  model.add(Dense(256, activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation = 'softmax'))

  print(model.summary())
  
  if not os.path.isdir(graph_dir): # Check if directory exists
    os.mkdir(graph_dir) # Create dir
    print ('Pasta criada com sucesso!')
	
  # Save Summary into file
  with open(log_Summary,'w') as fh:
	# Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
  
  #Save a nice img of model
  plot_model(model, to_file = img_model)
 
  model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

  checkpointer = ModelCheckpoint('model_lucaslb.h5', save_best_only=True, monitor='val_loss', mode='min')
						
  # Train the model
  #hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
  hist = model.fit(x_train, y_train, validation_split=0.25, callbacks=[checkpointer], epochs=epochs, batch_size=batch_size)
  
  # Plot and save Graphs
  plotGraph(hist, epochs)
  
  # Evaluate the model
  scores = model.evaluate(x_test, y_test)
  print("Accuracy: %.2f%%" % (scores[1]*100))
  
if __name__ == "__main__":
  
  import sys
  from pathlib import Path
  
  trainModel()