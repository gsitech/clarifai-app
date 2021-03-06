from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from PIL import Image
import requests, argparse

LABELS_DICT = {}
LABEL_CTR = 0

def url_download_to_image(file, url):
    '''Downloads a file from the url and correctly converts it to an image.
    
       :param file- A path to create a new image file
       :param url- url to download from
    '''
    with open(file, 'wb') as handle:
        r = requests.get(url, stream=True)
        for block in r.iter_content(1024):
            handle.write(block)

def check_if_image(img_path):
    '''Checks if the file is a valid image.
       
       :param img- absolute path to an image file
    '''
    try:
        Image.open(img_path)
    except IOError:
        print(img_path, "is not a valid image")
        exit(1)

def download_images(counter, url):
    '''Downloads images from a custom Clarifai database and
       obtains embeddings for each of them
    '''

    url_download_to_image('pic'+ str(counter)+'.jpg', url)

def skip_download_images(counter, url):
    pass

def _1024_embeddings(img, file, model):
    '''Obtains embeddings and writes them to a file handle
    '''
    f = model.predict([img])
    file.write(str(f['outputs'][0]['data']['embeddings'][0]['vector'])[1:-1] + '\n')

def get_labels(img, file1, file2, model):
    '''Obtains labels and writes them to a file handle
    '''
    global LABEL_CTR, LABELS_DICT
    f = model.predict([img])
    key = f['outputs'][0]['data']['concepts'][0]['name']
    
    if key not in LABELS_DICT.keys():
        LABELS_DICT[key] = LABEL_CTR
        LABEL_CTR+=1

    file1.write(str(LABELS_DICT[key]) + '\n')
    file2.write(str(key) + '\n')


def query_CLARIFAI(CLARIFAI_API_KEY, cmd='2', query_image = None):
    '''Queries Clarifai's API to obtain embeddings of images
    '''
    app = ClarifaiApp(api_key= CLARIFAI_API_KEY)
    img_num = 0

    counter = 0
    img_num+=1
    model_emb = app.models.get('general-v1.3', model_type='embed')
    model_lab = app.models.get('general-v1.3')

    func_dict = {'1':download_images, '2': skip_download_images}
    
    if __name__ == '__main__':
        print('Choose from the following options:')
        print('Enter "1" to download images from your custom database and obtain embeddings')
        print('Enter "2" to only obtain embeddings of images in your database')
        print('Enter "3" to obtain an embedding of a new query images')
        cmd = input('Choose either 1, 2 or 3: ')

        if cmd == '3':
            query_image = input('Enter the name of your query image')    

    with open('../data/embeddings.txt', 'w') as emb, open('../data/labels.txt', 'w') as lab, open('../data/labels_strings.txt', 'w') as lab_s:

        if cmd == '1' or cmd == '2':
            response = app.inputs.get_all()
            counter = 0
            for item in response:
                counter+=1
                print('Processing image ' + str(counter))
                func_dict[cmd](counter, item.url)
                _1024_embeddings(item, emb, model_emb)
                get_labels(item, lab, lab_s, model_lab)

        elif query_image!=None:
            check_if_image(query_image)
            _1024_embeddings(ClImage(filename= query_image), emb, model_emb)
            get_labels(item, lab, lab_s, model_lab)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Obtain the representation of an image in 1024 dimensions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='For further questions see the README'
    )

    parser.add_argument(
        'API_KEY',
        help='Your Clarifai API KEY.'
    )

    args = parser.parse_args()

    query_CLARIFAI(args.API_KEY)