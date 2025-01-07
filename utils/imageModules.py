from rembg import remove
import glob
import os
from PIL import Image
import cv2
import io


def remove_background(input_path, output_path, file_name_suffix = 'nobg'):
    in_files = []
    if os.path.isfile(input_path):
        in_files = [input_path] 
    elif os.path.isdir(input_path):
        in_files = glob.glob(input_path+'/*')
    else:
        print('Invalid input path:', input_path)
        #return
    print('Removing background from images at:', str(in_files))
    for file in in_files:
        with open(file, 'rb') as i:
            fileName, fileExtension = os.path.splitext(file)
            with open(output_path+'//'+os.path.basename(fileName)+"_"+file_name_suffix+fileExtension, 'wb') as o:
                input = i.read()
                output = remove(input)
                o.write(output)
    print ('Background removed images saved at:', output_path)


def remove_background_raw(imageData):
        output = remove(imageData)        
        return output 

def white_background_raw(imageData):
    original_image = Image.open(io.BytesIO(imageData))
    
    output_nobg = Image.open(io.BytesIO(remove_background_raw(imageData)))
    
    white_background = Image.new("RGB", original_image.size, "white")
    combined_image = Image.alpha_composite(white_background.convert("RGBA"), output_nobg.convert("RGBA"))
    
    # Convert the result back to RGB (if needed)
    combined_image = combined_image.convert("RGB")
    
    return combined_image

def white_background(input_path, output_path,):
    in_files = []
    if os.path.isfile(input_path):
        in_files = [input_path] 
    elif os.path.isdir(input_path):
        in_files = glob.glob(input_path+'/*')
    else:
        print('Invalid input path:', input_path)

    for file in in_files:
        with open(file, 'rb') as i:
        #    with open(output_path+'//'+os.path.basename(file), 'wb') as o:
            #original_image_bytes = i.read()
            original_nogb = remove(Image.open(file)) # Remove the background from the image return the datatype that is passed
            original_image = Image.open(file)
            #o.write(output)
            fileName, fileExtension = os.path.splitext(file)
            output_file = output_path+'//'+os.path.basename(fileName)+'_wbg'+fileExtension
            #original_image = Image.open(filePath)
            
            # Create a white background image with the same size as the original image
            white_background = Image.new("RGB", original_image.size, "white")
            
            # Paste the original image onto the white background
            combined_image = Image.alpha_composite(white_background.convert("RGBA"), original_nogb.convert("RGBA"))
            
            # Convert the result back to RGB (if needed)
            combined_image = combined_image.convert("RGB")
            combined_image.save(output_file)
            #print (output_file)

def resize_with_pad(input_path, output_path, target_width, target_height):
    '''
    Resize PIL image keeping ratio and using white background.
    input_path,
    output_path,
    target_width, 
    target_height
    '''
    in_files = []
    if os.path.isfile(input_path):
        in_files = [input_path] 
    elif os.path.isdir(input_path):
        in_files = glob.glob(input_path+'/*')
    else:
        print('Invalid input path:', input_path)

    target_ratio = target_height / target_width
    for file in in_files:
        #with open(file, 'rb') as i
        im = Image.open(file)
        fileName, fileExtension = os.path.splitext(file)
        output_file = output_path+'//'+os.path.basename(fileName)+'_pad'+fileExtension

        im_ratio = im.height / im.width
        if target_ratio > im_ratio:
            # It must be fixed by width
            resize_width = target_width
            resize_height = round(resize_width * im_ratio)
        else:
            # Fixed by height
            resize_height = target_height
            resize_width = round(resize_height / im_ratio)

        image_resize = im.resize((resize_width, resize_height), Image.LANCZOS )
        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
        background.paste(image_resize, offset)
        background.convert('RGB').save(output_file)
    print (f'Resiszed to {target_height}, {target_width}  and images saved at:', output_path)
    
def get_face_bbox(image_path):
    '''
    Get the bounding box of the face in the image
    '''
    import glob
    files = glob.glob(image_path+'/*')
    return_dic = {}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    for file in files:
        image = cv2.imread(file)
        faces = face_cascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            box  = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0))
            #print('ran')
        return_dic[file] = {'boundary':faces, 'mergedImage':  cv2.cvtColor(box,cv2.COLOR_BGR2RGB)}
    
    return return_dic
        

def crop_top_center(input_path, output_path, cropLength=512):
    
    import glob
    files = glob.glob(input_path+'/*')
    
    for file in files:
        image = cv2.imread(file)  
        ## Convert to PIL image
        image2 = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        box = ( max(0,int(image2.width//2-cropLength/2)), 0,min(image2.width,int(image2.width//2+512/2)) ,image2.height//2) 
        
        cropped_image = image2.crop(box)
        cropped_image.save(output_path+'//'+os.path.basename(file))
    print ('Cropped images saved at:', output_path)
        