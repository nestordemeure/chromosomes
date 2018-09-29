import cv2  # image manipulation (python opencv)
import numpy as np  # array manipulation
import glob  # path manipulation
import os  # folder creation

input_folder = "./png_unscaled_data/"
output_folder = "./png_scaled_contrast_data/"

new_height = 229
new_width = 103

def rescale_picture(picture, new_height, new_width):
    """takes a picture and produces a new picture of the given size
    fills the leftover with white"""
    height, width = picture.shape
    height_translation = (new_height - height) / 2
    width_translation = (new_width - width) / 2
    translation = np.float32([[1, 0, width_translation], [0, 1, height_translation]])
    white = (255, 255, 255)
    return cv2.warpAffine(picture, translation, dsize=(new_width, new_height), borderValue=white)

def adjust_contrast(picture):
    """adjust the contrast of an image by equalizing its histogram"""
    return cv2.equalizeHist(picture)

# iterates on all subfolders in the input folder
os.makedirs(output_folder)
for folder_path in glob.glob(input_folder + "*/"):
    print("Processing folder " + folder_path, flush=True)
    # create the output folder
    output_subfolder = folder_path.replace(input_folder, output_folder)
    os.makedirs(output_subfolder)
    # iterates on all png in the folder
    for input_path in glob.glob(folder_path + "*.png"):
        picture = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        new_picture = rescale_picture(adjust_contrast(picture), new_height, new_width)
        output_path = input_path.replace(input_folder, output_folder)
        cv2.imwrite(output_path, new_picture)

print("Finished with process.")