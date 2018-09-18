from clarifai.rest import ClarifaiApp
import os, requests, time, sys
from PIL import Image
import matplotlib.pyplot as plt

ROWS = 6
COLUMNS = 5
CLARIFAI_API_KEY = ''
PATH = ''

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

def url_download_to_image(file, url):
    '''Downloads a file from the url and correctly converts it to an image.
    
       :param file- A path to create a new image file
       :param url- url to download from
    '''
    with open(file, 'wb') as handle:
        r = requests.get(url, stream=True)
        for block in r.iter_content(1024):
            handle.write(block)

def add_image_to_subplot(img_path, index, title, plot):
    '''Adds an image to the subplot.
       
       :param img_path- A path to an image to add to the subplot
       :param index- The index within the subplot to place the image
       :param title- Title of the image
       :param plot- Outer canvas
    '''
    disp_image = Image.open(img_path).convert('RGBA')
    a = plot.add_subplot(ROWS, COLUMNS, index)
    a.title.set_text(title)
    a.axis('off')
    plt.imshow(disp_image)

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
        
else:
    CLARIFAI_API_KEY=sys.argv[1]
    PATH = sys.argv[2]

app = ClarifaiApp(api_key= CLARIFAI_API_KEY)
img_num = 0
while 1:
    if len(os.listdir(PATH)) > img_num and  check_if_image(os.path.join(PATH,os.listdir(PATH)[img_num])):
        fig=plt.figure(figsize=(8, 8))
        img = os.listdir(PATH)[img_num]
        response = app.inputs.search_by_image(filename = os.path.join(PATH,img))
        counter = 0
        i = 2
        cnt =0
        img_num+=1
        img_list=[]
        for _ in range (10):
            response.remove(response[-1])
        for item in response:
            counter+=1
            print(str(counter) + ") URL: " + item.url + ", SCORE: " + str(item.score) + ",  Metadata: " + str(item.metadata))
            url_download_to_image('pic'+ str(counter)+'.jpg', item.url)
            add_image_to_subplot(os.path.join(os.getcwd(),'pic'+ str(counter)+'.jpg'), cnt+i*COLUMNS+1, 'Score= ' + str(item.score)[2:4] + '%', fig)
            
            if cnt == COLUMNS-1:
                cnt=0
                i*=2
            else:
                cnt+=1
        
        add_image_to_subplot(os.path.join(PATH,img), COLUMNS/2 +1, 'Query Image', fig)

        plt.show()
        for i in range (10):
            os.remove(os.path.join(os.getcwd(),'pic'+ str(i+1)+'.jpg'))

        cmd = input('Would you like to delete the query image?(y/n): ')
        if cmd == 'y':
            os.remove(os.path.join(PATH,img))
            img_num-=1

    else:
        print("No image files found in PATH. Please add an image")
        time.sleep(10)

    command = input('Type exit to stop execution or just press Enter to search with the next image in the directory: ')
    if command == 'exit':
        exit(0)