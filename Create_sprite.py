import os, requests, time, argparse
from PIL import Image
import matplotlib.pyplot as plt
from model.utils import Params

#TODO
#Please compute the values for these variables
#Based on ceil(sqrt(Test_Data_Size))
ROWS = 15
COLUMNS = 15

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

def add_image_to_canvas(im, image, counter_x, counter_y, img_dim):
    image = Image.open(image, mode='r')
    image = image.resize((img_dim,img_dim), Image.ANTIALIAS)
    im.paste(image, box=(counter_x*img_dim,counter_y*img_dim), mask=None)

def cmp(a):
    return int(a[3:].split('.')[0])

parser = argparse.ArgumentParser()
parser.add_argument('image_dir',
                    help="Directory containing the images to be used for the sprite")
parser.add_argument('--sprite_save_dir', default='experiments/clarifai_sprite.png',
                    help="Directory to save the sprite image")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    #Create sprite
    counter_x = 0
    counter_y = 0
    im = Image.new(mode='RGB', size = (15*params.sprite_image_dim,15*params.sprite_image_dim))
    image_dir = sorted(os.listdir(args.image_dir), key=cmp)
    for image in image_dir:
        if  check_if_image(os.path.join(args.image_dir,image)):
            add_image_to_canvas(im, os.path.join(args.image_dir,image), counter_x, counter_y, params.sprite_image_dim)
            counter_x+=1
            if counter_x%15 == 0:
                counter_x=0
                counter_y+=1
            
    im.show()
    im.save(os.path.join(os.getcwd(),args.sprite_save_dir), format = 'PNG')