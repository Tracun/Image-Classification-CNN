import matplotlib.pyplot as plt
from random import shuffle
from PIL import Image
from array import *
import os

def header(width, height):
  size = width*height

  header = 'label'
  for x in range(1,size+1):
    header += ',pixel' + str(x)
  return header

def toCsv():
  # Load from and save to
  # Names[[FolderName, Filename]]
  names = [['training-images','train'], ['test-images','test']]
  
  # For resize image
  width = 64
  height = 64
  count = 1
  
  for name in names:

      file = 'output/' + str(name[1]) + '.csv'
      if (os.path.exists(file)):
          os.remove(file)
          print('Arquivo anterior deletado...')

      # Open a file to write the pixel data
      csvFile = open(file, 'w+')
      
      # Create the header line from width and height
      csvFile.write(header(width, height))
      csvFile.write('\n')

      data_image = array('B')
      data_label = array('B')

      FileList = []
      for dirname in os.listdir(name[0]):
          path = os.path.join(name[0], dirname)
          for filename in os.listdir(path):
              if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                  FileList.append(os.path.join(name[0], dirname, filename))

      shuffle(FileList)  # Usefull for further segmenting the validation set
      
      for filename in FileList:
		
        print('filename', filename)
        print('Quant. Imagens', count)
        count += 1
        label = str(filename.split('/')[1])
		
        # Translating a color image to black and white (mode 'L')
        Im = Image.open(filename).convert('L')
        ImResize = Im.resize((width, height), Image.ANTIALIAS)

        # Load the pixel info
        pixel = ImResize.load()

        # Add the label 
        imgArray = '{0}'.format(label)

        # Read the details of each pixel and write them to the file
        # Each line is an image
        for x in range(width):
          for y in range(height):
            imgArray += ',{0}'.format(pixel[y, x])
        csvFile.write(imgArray)
        csvFile.write('\n')
      num = 'test'
      csvFile.close()
  print('Done !!!')
  
if __name__ == "__main__":
  toCsv()
  
  