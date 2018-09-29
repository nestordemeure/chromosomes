import cv2  # image manipulation (python opencv)
import glob  # path manipulation

input_folder = "./png_unscaled_data/"

max_height = 0
max_width = 0
file_number = 0

# iterates on all subfolders in the input folder
for folder_path in glob.glob(input_folder + "*/"):
    print("Processing folder " + folder_path, end='', flush=True)
    # iterates on all png in the folder
    # TODO we could be more efficient and work in parallel
    for input_path in glob.glob(folder_path + "*.png"):
        picture = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # TODO we could be more efficient and only load the size
        height, width = picture.shape
        max_height = max(height, max_height)
        max_width = max(width, max_width)
        file_number += 1
    print(", height:{h} width:{w} ({n} files read)".format(h=max_height, w=max_width, n=file_number), flush=True)

# height:103 width:229 (153345 files read)
print("All files have been processed, height:{h} width:{w} ({n} files read)".format(h=max_height, w=max_width, n=file_number))
