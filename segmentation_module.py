import os
from typing import Tuple, List
import logging
import cv2
from PIL import Image
from functools import reduce
import cv2
import os
from PIL import Image
import numpy as np
import mediapipe as mp
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
import os

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MASK_OPTION_0_BACKGROUND = 'background'
MASK_OPTION_1_HAIR = 'hair'
MASK_OPTION_2_BODY = 'body (skin)'
MASK_OPTION_3_FACE = 'face (skin)'
MASK_OPTION_4_CLOTHES = 'clothes'
MASK_OPTION_5_ACCESSORIES = 'others/Accessories'
model_path_ext = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
#'./models_linker/raw_models/mediapipe/mutliclass_image_seg/selfie_multiclass_256x256.tflite'
mask_targets = ['clothes','background']
mask_dilation = 0

import urllib
import glob
#IMAGE_FILENAMES = ['segmentation_input_rotation0.jpg']

#for name in IMAGE_FILENAMES:
url =model_path_ext
model_loc = './model/'

model_path =model_loc +'selfie_multiclass_256x256.tflite'
if os.path.exists(model_path):
    pass
else:
    print('Downloading models')
    os.makedirs(model_loc)
    urllib.request.urlretrieve(url, model_path)



def get_mediapipe_image( image: Image) -> mp.Image:
    # Convert gr.Image to NumPy array
    numpy_image = np.asarray(image)

    image_format = mp.ImageFormat.SRGB

    # Convert BGR to RGB (if necessary)
    if numpy_image.shape[-1] == 4:
        image_format = mp.ImageFormat.SRGBA
    elif numpy_image.shape[-1] == 3:
        image_format = mp.ImageFormat.SRGB
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=image_format, data=numpy_image)

def transform_image(image_source):
    transform = T.Compose(
        [
            F.resize(image_source.size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:

    image_source = Image.open(image_path).convert("RGB")
    
    return transform_image(image_source)
    


def perform_segmentation(file, raw_data = False):

    
    if raw_data:
        
        logging.info('Reading raw image file')
        image = file
    else:
        logging.info('Opeeing image file from'+str(file))
        image = Image.open(file)


    options = ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=model_path),
                                                running_mode=VisionRunningMode.IMAGE,
                                                output_category_mask=True)
    # Create the image segmenter
    
    with ImageSegmenter.create_from_options(options) as segmenter:
    
        # Retrieve the masks for the segmented image
        media_pipe_image = get_mediapipe_image(image=image)
        segmented_masks = segmenter.segment(media_pipe_image)
        masks = []
        for i, target in enumerate(mask_targets):
            # https://developers.google.com/mediapipe/solutions/vision/image_segmenter#multiclass-model
            # 0 - background
            # 1 - hair
            # 2 - body - skin
            # 3 - face - skin
            # 4 - clothes
            # 5 - others(accessories)
            mask_index = 0
            if target == MASK_OPTION_1_HAIR:
                mask_index = 1
            if target == MASK_OPTION_2_BODY:
                mask_index = 2
            if target == MASK_OPTION_3_FACE:
                mask_index = 3
            if target == MASK_OPTION_4_CLOTHES:
                mask_index = 4
            if target == MASK_OPTION_5_ACCESSORIES:
                mask_index = 5
    
            masks.append(segmented_masks.confidence_masks[mask_index])
    
        image_data = media_pipe_image.numpy_view()
        image_shape = image_data.shape
    
        # convert the image shape from "rgb" to "rgba" aka add the alpha channel
        if image_shape[-1] == 3:
            image_shape = (image_shape[0], image_shape[1], 4)
    
        mask_background_array = np.zeros(image_shape, dtype=np.uint8)
        mask_background_array[:] = (0, 0, 0, 255)
    
        mask_foreground_array = np.zeros(image_shape, dtype=np.uint8)
        mask_foreground_array[:] = (255, 255, 255, 255)
    
        mask_arrays = []
        for i, mask in enumerate(masks):
            condition = np.stack((mask.numpy_view(),) * image_shape[-1], axis=-1) > 0.25
            mask_array = np.where(condition, mask_foreground_array, mask_background_array)
            mask_arrays.append(mask_array)
    
        # Merge our masks taking the maximum from each
        merged_mask_arrays = reduce(np.maximum, mask_arrays)
    
        # Dilate or erode the mask
        if mask_dilation > 0:
            merged_mask_arrays = cv2.dilate(merged_mask_arrays, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilation + 1, 2*mask_dilation + 1), (mask_dilation, mask_dilation)))
        elif mask_dilation < 0:
            merged_mask_arrays = cv2.erode(merged_mask_arrays, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_dilation + 1, 2*mask_dilation + 1), (mask_dilation, mask_dilation)))
    
        # Create the image
        mask_image = Image.fromarray(merged_mask_arrays)
        print(mask_image)
        return mask_image