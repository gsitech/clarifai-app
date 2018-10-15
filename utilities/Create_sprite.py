import os, requests, time, argparse
from PIL import Image
import matplotlib.pyplot as plt

#TODO
#Please compute the values for these variables
#Based on ceil(sqrt(Dataset_Size))
ROWS = 15
COLUMNS = 15
IMG_DIM = 74

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

def add_image_to_canvas(im, image, counter_x, counter_y):
    image = Image.open(image, mode='r')
    image = image.resize((IMG_DIM,IMG_DIM), Image.ANTIALIAS)
    im.paste(image, box=(counter_x*IMG_DIM,counter_y*IMG_DIM), mask=None)

def cmp(a):
    return int(a[3:].split('.')[0])

parser = argparse.ArgumentParser()
parser.add_argument('image_dir',
                    help="Directory containing the images to be used for the sprite")
parser.add_argument('--sprite_save_dir', default='../data/clarifai_sprite.png',
                    help="Directory to save the sprite image")
parser.add_argument('--model_dir', default='../experiments/base_model',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    #Create sprite
    counter_x = 0
    counter_y = 0
    im = Image.new(mode='RGB', size = (15*IMG_DIM,15*IMG_DIM))
    image_dir = sorted(os.listdir(args.image_dir), key=cmp)
    for image in image_dir:
        if  check_if_image(os.path.join(args.image_dir,image)):
            add_image_to_canvas(im, os.path.join(args.image_dir,image), counter_x, counter_y)
            counter_x+=1
            if counter_x%15 == 0:
                counter_x=0
                counter_y+=1
            
    im.show()
    im.save(os.path.join(os.getcwd(),args.sprite_save_dir), format = 'PNG')