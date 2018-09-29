import cv2  # python opencv image import, export and transformation
import numpy as np  # array manipulation

input_folder = "./png_unscaled_data/1/"
output_folder = "./"

#--------------------------------------------------------------------------------------------------
# IMPORTATION

picture = cv2.imread(input_folder + "16248-2_1_118585.png", cv2.IMREAD_GRAYSCALE)

#--------------------------------------------------------------------------------------------------
# TRANSFORMATION

def rescale_picture(picture, new_height, new_width):
    """takes a picture and produces a new picture of the given size
    fills the leftover with white"""
    height = picture.shape[0]
    width = picture.shape[1]
    height_translation = (new_height - height) / 2
    width_translation = (new_width - width) / 2
    translation = np.float32([[1, 0, width_translation], [0, 1, height_translation]])
    white = (255, 255, 255)
    return cv2.warpAffine(picture, translation, dsize=(new_height, new_width), borderValue=white)

def adjust_contrast(picture):
    """adjust the contrast of an image by equalizing its histogram"""
    return cv2.equalizeHist(picture)

new_picture = rescale_picture(adjust_contrast(picture), 100, 100)

#--------------------------------------------------------------------------------------------------
# EXPORTATION

# export modified image
cv2.imwrite(output_folder + "input.png", picture)
cv2.imwrite(output_folder + "output.png", new_picture)
