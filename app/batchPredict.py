import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels

#size of image
width = 64
height = 64

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('confusion_matrix: ', cm)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def confusionMatrix(y_true, y_pred, classes_names):
  cMatrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

  #make a nice plot
  print('confusion_matrix: ', cMatrix)
  
def predictTests(model, tests):

  from keras.models import load_model
  
  #CSV that contains images to test the trained model
  test = pd.read_csv(tests)
  model = load_model(model)
  test_labels = test['label']
  
  #Remove as equal classes, leaving 20 classes (20 different images)
  labels = test['label'].values
  
  unique_val = np.array(labels)
  #Unique labels like [0,1,2,3,...]
  class_names = np.unique(unique_val)

  test.drop('label', axis = 1, inplace = True)

  test_images = test.values
  test_images = np.array([np.reshape(i, (width, height)) for i in test_images])
  test_images = np.array([i.flatten() for i in test_images])

  #Transforms labels into an array of classes (24 classes) per 8566 rows. The 0 does not belong to the classes, 1 belongs to the class
  label_binrizer = LabelBinarizer()
  
  #test_labels = y_True
  test_labels = label_binrizer.fit_transform(test_labels)

  test_images = test_images.reshape(test_images.shape[0], width, height, 1)

  print('test_images.shape: ', test_images.shape)

  #Resizes the array to 4 dimensions
  imgToPredict = test_images[0].reshape(1,width,height,1)
  print('imgToPredict.shape: ', imgToPredict.shape)

  #Realiza a predicao
  y_pred = model.predict(test_images)
  
  #Mostra a porcentagem de acerto com o dataset de testes
  from sklearn.metrics import accuracy_score
  print('accuracy_score: ', accuracy_score(test_labels, y_pred.round()))
  
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  plot_confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1), classes=class_names,
                        title='Confusion matrix, without normalization')
  
  plt.savefig('./Train_Graphs_Logs/nonNormalizedConfusionMatrix.png')
  plt.close()
  
  # Plot normalized confusion matrix
  plot_confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1), classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
  plt.savefig('./Train_Graphs_Logs/normalizedConfusionMatrix.png')
  
  plt.show()
  plt.close()

  return y_pred
  
if __name__ == "__main__":
  
  import sys
  from pathlib import Path
  
  param = sys.argv[1:]
  print('param[0]: ', param[0])
	
  #param[0] ---> Modelo .H5
  #param[1] ---> .CSV contendo imagens para predicao
  predictTests(param[0], param[1])
  
  
  
  
  