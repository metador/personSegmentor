import base64
import json
import logging
import multiprocessing
import os 
from flask import Flask, render_template, request, jsonify
#from tasks import perform_image_segmentation
from segmentation_module import perform_segmentation
from base64 import b64encode
import io
import time
from PIL import Image
from datetime import datetime
import requests
import threading
## https://chat.openai.com/share/3db77dd2-5467-4a07-b3ca-8496f9ff3d7b

app = Flask(__name__)
image_path = './uploads/'
image_api_path = './uploads/api/'
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    
    # Save the uploaded image
    image_path = './uploads/input_image.jpg'
    image_file.save(image_path)

    start = time.time()
    # Perform image segmentation asynchronously using Celery
    mask = perform_segmentation(image_path).convert('RGB')
    
    data = io.BytesIO()
    mask.save(data, "JPEG")
    encoded_img_data = b64encode(data.getvalue())
    data = 'data:image/png;base64,' + encoded_img_data.decode('ascii')
    #task = perform_image_segmentation.delay(image_path)

    #return jsonify({'task_id': task.id})
    #og_image= 'data:image/png;base64,' + encoded_img_data.decode('ascii')
    end = time.time()
    duration  = end-start
    return f"""<img src={data}></img><img src={image_path}></img> Time taken {duration}"""


def save_image_async(image_data, image_path):
    with open(image_path, 'wb') as f:
        f.write(image_data)
    
        print('Image saved at:', image_path)
    return None


@app.route('/saveimage', methods=['POST'])
def save_image_api():
    data = request.get_json()
    image_file = data.get('image')
    ##print(image_file)
    #image_data = io.BytesIO(base64.b64decode(image_file))
    if 'https' in image_file:
        image_file = base64.b64encode(requests.get(image_file).content).decode('utf-8')
    
    image_path =  os.path.join(image_api_path,'api_image_'+ str(datetime.now().strftime("%Y%m%d.%H.%M.%S%f")) +'.png')
    with open(image_path, 'wb') as f:
        f.write(base64.b64decode(image_file))
    return {'status': 'success', 'image_path': image_path}

@app.route('/segmentapi', methods=['POST'])
def segment_api():
    # if 'image' not in request:
    #     return
    #
    #   jsonify({'error': 'No image provided'})
    #Logger.debug('This is the message', str(request.files))
    #image_file = request.files['image']  # body['image']
    #Logger.info(image_file)

    ## Getting data request from api sent as body: JSON.stringify({ image: base64Image }),}
    data = request.get_json()
    image_file = data.get('image')
    start = time.time()

    # Decode base64 image
    image_data = io.BytesIO(base64.b64decode(image_file))
    # Save the uploaded image
    
    p = threading.Thread(target=save_image_async, args=(base64.b64decode(image_file), os.path.join(image_api_path, 'original_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
    #multiprocessing.Process(target=save_image_async, args=(base64.b64decode(image_file), os.path.join(image_api_path, 'original_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
   
    
    print('Starting segmentation')
    #image_path = './uploads/input_image.jpg'
    #image_file.save(image_path)

    # Perform image segmentation asynchronously using Celery
    #logging.info(msg = image_data)
    mask = perform_segmentation(image_data, raw_data = False).convert('RGB')
    byte_arr = io.BytesIO()
    mask.save(byte_arr,format='png')

    # Get the byte value from the BytesIO object
    byte_arr = byte_arr.getvalue()
    encoded_img_data = b64encode(byte_arr)
    data =  encoded_img_data.decode('utf-8')
    #task = perform_image_segmentation.delay(image_path)


    end = time.time()
    duration  = end-start
    p.daemon = True
    p.start()
    ## Create a new non blocking process to save the image
    p2 = threading.Thread(target=save_image_async, args=(byte_arr, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
    
    #p2 = multiprocessing.Process(target=save_image_async, args=(byte_arr, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
    p2.daemon = True
    p2.start()

    #save_image(byte_arr, os.path.join(image_api_path,'mask_image'+str(time.time())+'.png'))
    return jsonify({'image': data, 'time': duration})
    



if __name__ == '__main__':
    app.run(debug=True)