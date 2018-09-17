from clarifai.rest import ClarifaiApp
import os, requests, time, sys
from PIL import Image
import matplotlib.pyplot as plt

CLARIFAI_API_KEY = 'a28786feb1b64f4193ece897e65d4fcb'
PATH = 'C:\\Users\\VV\\Documents\\My Bluetooth\\Clarifai'

if len(sys.argv) < 2: 
    if len(CLARIFAI_API_KEY)==0 :
        print('Please provide a key')
        exit(1)
    elif len(PATH)==0:
        print('Please provide a PATH to the folder containing the query image')
        exit(1)

app = ClarifaiApp(api_key= CLARIFAI_API_KEY)

rows = 6
columns = 5

while 1:
    if len(os.listdir(PATH)) > 0:
        fig=plt.figure(figsize=(8, 8))
        img = os.listdir(PATH)[0]
        response = app.inputs.search_by_image(filename = os.path.join(PATH,img))
        counter = 0
        i = 2
        cnt =0
        img_list=[]
        for _ in range (10):
            response.remove(response[-1])
        for item in response:
            counter+=1
            print(str(counter) + ") URL: " + item.url + ", SCORE: " + str(item.score) + ",  Metadata: " + str(item.metadata))
            with open('pic'+ str(counter)+'.jpg', 'wb') as handle:
                r = requests.get(item.url, stream=True)
                for block in r.iter_content(1024):
                        handle.write(block)

            disp_image = Image.open(os.path.join(os.getcwd(),'pic'+ str(counter)+'.jpg')).convert('RGBA')
            a = fig.add_subplot(rows, columns, cnt+i*columns+1)
            a.title.set_text('Score= ' + str(item.score)[2:4] + '%')
            a.axis('off')
            plt.imshow(disp_image)
            if cnt == columns-1:
                cnt=0
                i*=2
            else:
                cnt+=1
        
        a = fig.add_subplot(rows, columns, (columns/2 +1))
        a.title.set_text('Query Image')
        a.axis('off')
        plt.imshow(Image.open(os.path.join(PATH,img)))
        plt.show()
        for i in range (10):
            os.remove(os.path.join(os.getcwd(),'pic'+ str(i+1)+'.jpg'))

        os.remove(os.path.join(PATH,img))

    input('Please send a new image via Bluetooth and then press enter')
    time.sleep(10)
