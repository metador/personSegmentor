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
from utils.imageModules import remove_background_raw,white_background_raw
import requests
import threading
import gc

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
    tid = data.get('tid')
    start = time.time()

    # Decode base64 image inside a context manager
    with io.BytesIO(base64.b64decode(image_file)) as image_data:
        # Start save thread before heavy processing
        save_thread = threading.Thread(
            target=save_image_async,
            args=(base64.b64decode(image_file), 
                    os.path.join(image_api_path, f'original_image_{datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")}.png'))
        )
        save_thread.daemon = True
        save_thread.start()

        print('Starting segmentation tid:', tid)
        
        # Perform segmentation and immediately convert to RGB
        mask = perform_segmentation(image_data, raw_data=False).convert('RGB')

    
    
  # Use context manager for byte array operations
    with io.BytesIO() as byte_arr:
        mask.save(byte_arr, format='png')
        byte_data = byte_arr.getvalue()
        encoded_img_data = b64encode(byte_data)
        result_data = encoded_img_data.decode('utf-8')

    #task = perform_image_segmentation.delay(image_path)

    # Clear the mask from memory
    mask.close()
    del mask

    end = time.time()
    duration  = end-start

    ## Create a new non blocking process to save the image
    save_thread2 = threading.Thread(target=save_image_async, args=(byte_data, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
    
    #p2 = multiprocessing.Process(target=save_image_async, args=(byte_arr, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
    save_thread2.daemon = True
    save_thread2.start()
    gc.collect()
    #save_image(byte_arr, os.path.join(image_api_path,'mask_image'+str(time.time())+'.png'))
    return jsonify({'image': result_data, 'time': duration})


@app.route('/rmbg', methods=['POST'])
def remove_background_api():
    #if 'image' not in request:
    #    returnjsonify({'error': 'No image provided'})
    #Logger.debug('This is the message', str(request.files))
    #image_file = request.files['image']  # body['image']
    #Logger.info(image_file)

    ## Getting data request from api sent as body: JSON.stringify({ image: base64Image }),}
    data = request.get_json()
    image_file = data.get('image')
    start = time.time()
    
    print('Starting Removing Background')
    #image_path = './uploads/input_image.jpg'
    #image_file.save(image_path)

    # Perform image segmentation asynchronously using Celery
    #logging.info(msg = image_data)
    image_data_noBg = remove_background_raw(base64.b64decode(image_file))
    #mask = perform_segmentation(io.BytesIO(image_data_noBg), raw_data = False).convert('RGB')
    #mask = perform_segmentation(image_data, raw_data = False).convert('RGB')
    # byte_arr = io.BytesIO()
    # image_data_noBg.save(byte_arr,format='png')

    # # Get the byte value from the BytesIO object
    # byte_arr = byte_arr.getvalue()
    # encoded_img_data = b64encode(byte_arr)
    # data =  encoded_img_data.decode('utf-8')
    #task = perform_image_segmentation.delay(image_path)

    image_path = './uploads/input_image_nobg.jpg'
    with open(image_path, 'wb') as o:
        #input = i.read()
        #output = remove(input, force_return_bytes=True)
        o.write(image_data_noBg)    
    encoded_img_data = b64encode(image_data_noBg)
    data =  encoded_img_data.decode('utf-8')
    end = time.time()
    duration  = end-start


    #save_image(byte_arr, os.path.join(image_api_path,'mask_image'+str(time.time())+'.png'))
    return jsonify({'image': data, 'time': duration})


@app.route('/whitebg', methods=['POST'])
def white_background_api():
    # if 'image' not in request:
        
    #     return jsonify({'error': 'No image provided'})

    ## Getting data request from api sent as body: JSON.stringify({ image: base64Image }),}
    data = request.get_json()
    image_file = data.get('image')
    start = time.time()
    
    print('Starting change to white Background')
    #image_path = './uploads/input_image.jpg'
    #image_file.save(image_path)

    # Perform image segmentation asynchronously using Celery
    #logging.info(msg = image_data)
    image_data_whiteBg = white_background_raw(base64.b64decode(image_file))
  
    image_path = './uploads/input_image_whitebg.jpg'
    image_data_whiteBg.save(image_path)
    
    byte_arr = io.BytesIO()
    image_data_whiteBg.save(byte_arr,format='png')

    # Get the byte value from the BytesIO object
    byte_arr = byte_arr.getvalue()
    encoded_img_data = b64encode(byte_arr)
    data =  encoded_img_data.decode('utf-8')
    
    #encoded_img_data = b64encode(image_data_whiteBg)
    #data =  encoded_img_data.decode('utf-8')
    end = time.time()
    duration  = end-start


    #save_image(byte_arr, os.path.join(image_api_path,'mask_image'+str(time.time())+'.png'))
    return jsonify({'image': data, 'time': duration})




def get_options(option):
    print(option)
    if option == 'Asian':
        return {'lora': '<lora:iu_v35:1>,', 'race': ''}
    if type(option)==list and 'Model' in option[0]:
        return {'lora': '<lora:randwgil2v6:1>, beautiful face,', 'race': ''}
    else:
        return {'lora': '', 'race': option + ' '}

def delay(ms):
    time.sleep(ms / 1000)


@app.route('/modelSwitchApi', methods=['POST'])
def modelswitch_api():
    # if 'image' not in request:
    #     return
    #
    #   jsonify({'error': 'No image provided'})
    #Logger.debug('This is the message', str(request.files))
    #image_file = request.files['image']  # body['image']
    #Logger.info(image_file)

    ## Getting data request from api sent as body: JSON.stringify({ image: base64Image }),}
    start = time.time()
    try:
        data = request.get_json()
        image_file = data.get('image')
        print(image_file)
        if image_file is None:
            return jsonify({'error': 'No image provided'})
        
        requestBody = request.get_json()
        #base64Image = requestBody['image']
        image_file #image file 
        option = requestBody['param']
        prompt_params = get_options(option)
        lora = prompt_params['lora']
        race = prompt_params['race']
 
    except Exception as e:
        print('Error:', e)
        return jsonify({'error': 'Internal Server Error, error in input'}), 500
    
        

    #save_image(byte_arr, os.path.join(image_api_path,'mask_image'+str(time.time())+'.png'))
    #return jsonify({'image': data, 'time': duration})\
        
        
    ### Start Genrating the image
    
    try:
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
        encoded_mask_data = b64encode(byte_arr)
        mask_data =  encoded_mask_data.decode('utf-8')
        #task = perform_image_segmentation.delay(image_path)


    
        p.daemon = True
        p.start()
        ## Create a new non blocking process to save the image
        p2 = threading.Thread(target=save_image_async, args=(byte_arr, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
        
        #p2 = multiprocessing.Process(target=save_image_async, args=(byte_arr, os.path.join(image_api_path, 'mask_image_' + str(datetime.now().strftime("%Y%m%d.%H.%M.%S.%f")) + '.png')))
        p2.daemon = True
        p2.start()
        
    
        # try:
        if True:
            # Parse the JSON body which contains the base64-encoded image data


            hostUrl = 'http://api.omniinfer.io/'
            img2imgurl = 'v2/img2img'
            progressUrl = 'v2/progress'
            url = hostUrl + img2imgurl


            payload = {
                "prompt": f"beautiful {race}girl, {lora} smiling girl, white background, Best quality, masterpiece, ultra high res, (photorealistic:1.4), (sharp focus)",
                "negative_prompt": "easynegative,(3d,non-symmetrical eyes,jewelry , render, cgi, doll, painting, fake, cartoon, 3d modeling:1.4), (worst quality, low quality:1.4), child, deformed, malformed, malformed face, bad teeth, bad hands, bad fingers, bad eyes, long body, blurry, duplicated, cloned, duplicate body parts, disfigured, extra limbs, fused fingers, extra fingers, twisted, distorted, malformed hands, mutated hands, mutated fingers, conjoined, missing limbs, bad anatomy, bad proportions, logo, watermark, text, copyright, signature, lowres, mutated, mutilated, artifacts, gross, ugly",
                "steps": 20,
                "batch_size": 1,
                "denoising_strength": 0.75,
                "init_images": [image_file],
                "inpaint_full_res": 1,
                "inpaint_full_res_padding": 0,
                "inpainting_fill": 1,
                "inpainting_mask_invert": 1,
                "mask_blur": 5,
                "do_not_save_samples": False,
                "mask":mask_data,  
                "restore_faces": True,
                "sampler_name": "DPM++ 2M Karras",
                "model_name": "realisticVisionV40_v40VAE-inpainting_81543.safetensors",
                "vae": "vae-ft-mse-840000-ema-pruned.safetensors"
            }

            headers = {
                'Content-Type': 'application/json',
                'X-Omni-Key': '80ba18f8-6366-4524-a72e-c375ed93c0b3'
            }

            externalApiResponse = requests.post(url, headers=headers, json=payload, timeout=540)

            if externalApiResponse.status_code != 200:
                raise Exception(f"HTTP error! Status: {externalApiResponse.status_code}")

            externalApiData = externalApiResponse.json()
            print(externalApiData)
            task_id = externalApiData['data']['task_id']
            pollStatus = 0
            pollUrl = f"{hostUrl}{progressUrl}?task_id={task_id}"

            while pollStatus == 0 or pollStatus == 1:
                delay(5000)
                omminfierPoll = requests.get(pollUrl, headers=headers, timeout=540)
                pollResponse = omminfierPoll.text
                print(pollResponse)
                parsedResponse = json.loads(pollResponse)
                pollStatus = parsedResponse['data']['status']
                
                duration  = time.time()-start
                if duration > 120:
                    return jsonify({'image': 'NoImage', 'time': duration,'status': 'ok' if pollStatus==2 else 'error'})  
        
            print( parsedResponse['data']['imgs'][0])
            print('status:'+str(pollStatus))
            return jsonify({'image': parsedResponse['data']['imgs'][0], 'time': duration,'status': 'ok' if pollStatus==2 else 'error'})  
    
    except Exception as e:
            print('Error:', e)
            return jsonify({'error': 'Internal Server Error, error in generation'}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)