import re
import os
import numpy as np

from PIL import Image

dumping_file_path = "C:\\temp\\normalized_image826885234_f_560.bin"


def regen_image(arr):
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    # Reverse normalization
    array = arr * std + mean

    # Clip the values to [0, 1] range
    array = np.clip(array, 0, 1)

    # Convert to [0, 255] range and uint8 type
    array = (array * 255).astype(np.uint8)
    return array


filename = os.path.basename(dumping_file_path)
res = re.search(r".+(\d+)_([u|f])_(\d+)", filename)
dtype = np.uint8 if res[2] == 'u' else np.float32
# load the binary raw data from the file
with open(dumping_file_path, 'rb') as file:
    raw_data = np.fromfile(file, dtype=dtype)


image_width = int(res[3])
image_height = int(raw_data.size / image_width) // 3
raw_data = raw_data.reshape((image_height, image_width, 3))

# from bgr to rgb
# raw_data = raw_data[:, :, ::-1]

# save the image to disk
if dtype == np.float32:
    raw_data = regen_image(raw_data)

img = Image.fromarray(raw_data)
img.save(dumping_file_path + ".png")
img.show()
