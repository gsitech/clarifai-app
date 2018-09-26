from clarifai.rest import ClarifaiApp
import os, requests, sys
from PIL import Image

CLARIFAI_API_KEY = ''
PATH = ''

def url_download_to_image(file, url):
    '''Downloads a file from the url and correctly converts it to an image.
    
       :param file- A path to create a new image file
       :param url- url to download from
    '''
    with open(file, 'wb') as handle:
        r = requests.get(url, stream=True)
        for block in r.iter_content(1024):
            handle.write(block)

def download_images_obtain_1024_vector():
    file = open('embeddings.txt', 'w')
    counter = 0
    for item in response:
        counter+=1
        print(str(counter) + ") URL: " + item.url + ", REGIONS: " + str(item.regions))
        url_download_to_image('pic'+ str(counter)+'.jpg', item.url)
        f = model.predict([item])
        file.write(str(f['outputs'][0]['data']['embeddings'][0]['vector']) + '\n')
        if counter == 4:
            break

    file.close()

def obtain_1024_vector():
    file = open(os.path.join(PATH,'embeddings.txt'))
    counter = 0
    for item in response:
        counter+=1
        print(str(counter) + ") URL: " + item.url + ", REGIONS: " + str(item.regions))
        f = model.predict([item])
        file.write(str(f['outputs'][0]['data']['embeddings'][0]['vector']) + '\n')

    file.close()
'''
if len(sys.argv) == 3: 
    CLARIFAI_API_KEY = sys.argv[1]
    PATH = sys.argv[2]
elif len(sys.argv) > 3:
    print('Too many arguments')
    exit(1)
else:
    if len(CLARIFAI_API_KEY)==0:
        print('Please provide a key')
        exit(1)
    elif len(PATH)==0:
        print('Please provide a PATH to the folder containing the query image')
        exit(1)
'''
#def query_CLARIFAI():
app = ClarifaiApp(api_key= CLARIFAI_API_KEY)
img_num = 0

response = app.inputs.get_all()

counter = 0
img_num+=1
model = app.models.get('general-v1.3', model_type='embed')
func_dict = {'y':download_images_obtain_1024_vector, 'n':obtain_1024_vector}
cmd = input('Would you want to download the database images? (y/n)')    

func_dict[cmd]()
