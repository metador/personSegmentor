import sys
import os

#print(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



from utils.imageModules import *

input_path = "D:\GDrive.melwinjamesp\Companies\\aodel\dataset_raw\meshop\MelDataSet\loras\white\person2"
output_path = "D:\GDrive.melwinjamesp\Companies\\aodel\dataset_raw\meshop\MelDataSet\loras\white\person2_ench2"


#resize_with_pad(input_path, output_path, 512, 512)

#input = "D:\GDrive.melwinjamesp\Companies\\aodel\dataset_raw\meshop\MelDataSet\loras\white\person2_ench2"
#white_background(input_path, output_path)
crop_top_center(input_path, output_path, 512)

input_path = "D:\GDrive.melwinjamesp\Companies\\aodel\dataset_raw\meshop\MelDataSet\loras\white\person2_ench2"

white_background(input_path, output_path)