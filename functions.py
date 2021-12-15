import copy
import numpy as np
from PIL import Image

def format_np_output(np_arr):
    
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
        
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
     
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
     
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
     
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def recreate_image(im_as_var):
     
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im
