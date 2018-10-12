from clarifai.rest import ClarifaiApp
import os, requests, time, sys
from PIL import Image
import matplotlib.pyplot as plt

ROWS = 15
COLUMNS = 15
PATH = '/home/shashankiyer/Clarifai/thumb'

def check_if_image(img_path):
    '''Checks if the file is a valid image.
       
       :param img- absolute path to an image file
       :return- True if valid image
    '''
    try:
        Image.open(img_path)
        return True
    except IOError:
        print(img_path, "is not a valid image")
        return False

def add_image_to_subplot(img_path, index, fig):
    '''Adds an image to the subplot.
       
       :param img_path- A path to an image to add to the subplot
       :param index- The index within the subplot to place the image
       :param title- Title of the image
       :param plot- Outer canvas
    '''
    disp_image = Image.open(img_path).convert('RGBA')
    a = fig.add_subplot(ROWS, COLUMNS, index)
    a.axis('off')
    plt.imshow(disp_image, aspect='auto', interpolation='nearest')

fig=plt.figure(figsize=(30, 30))

counter = 0

for image in os.listdir(PATH):
    if  check_if_image(os.path.join(PATH,image)):
        
        counter+=1
        
        add_image_to_subplot(os.path.join(PATH,image), counter, fig)
        
plt.subplots_adjust(hspace=0, wspace=0)

plt.savefig("clarifai_sprite.jpg", bbox_inches='tight', pad_inches=0)
plt.show()