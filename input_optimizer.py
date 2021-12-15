import os
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.optim import SGD
from torch.autograd import Variable
from functions import recreate_image, save_image

use_cuda = torch.cuda.is_available()

class RegularizedClassSpecificImageGeneration():

    def __init__(self, model, target_class):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model.cuda() if use_cuda else model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        
        self.created_image = np.uint8(np.random.uniform(0, 255, (512, 512, 3)))
        #import cv2
        #self.created_image = cv2.imread('../input_images/aaa.jpg')
        #self.created_image = cv2.resize(self.created_image, (256, 512))
        #self.created_image = np.transpose(self.created_image, (1, 0, 2))
        #cv2.imshow('image', self.created_image)
        
        if not os.path.exists(f'./generated/class_{self.target_class}'):
            os.makedirs(f'./generated/class_{self.target_class}')

    def generate(self, iterations=1000, blur_freq=4, blur_rad=1, wd=0.0001, clipping_value=0.1):
       
        initial_learning_rate = 20
        for i in range(1, iterations):
            # Process image and return variable

            if i % blur_freq == 0:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False, blur_rad)
            else:
                self.processed_image = preprocess_and_blur_image(
                    self.created_image, False)

            if use_cuda:
                self.processed_image = self.processed_image.cuda()

            # Define optimizer for the image - use weight decay to add regularization
            # in SGD, wd = 2 * L2 regularization (https://bbabenko.github.io/weight-decay/)
            optimizer = SGD([self.processed_image],
                            lr=initial_learning_rate, weight_decay=wd)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i in np.linspace(0, iterations, 10, dtype=int):
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.cpu().numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()

            if clipping_value:
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), clipping_value)
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu())
            a = np.linspace(1, iterations, 10, dtype=int)
            if i in np.linspace(1, iterations, 30, dtype=int):
                # Save image
                im_path = f'./generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
                save_image(self.created_image, im_path)

        #save final image
        im_path = f'./generated/class_{self.target_class}/c_{self.target_class}_iter_{i}_loss_{class_loss.data.cpu().numpy()}.jpg'
        save_image(self.created_image, im_path)

        
        with open(f'./generated/class_{self.target_class}/run_details.txt', 'w') as f:
            f.write(f'Iterations: {iterations}\n')
            f.write(f'Blur freq: {blur_freq}\n')
            f.write(f'Blur radius: {blur_rad}\n')
            f.write(f'Weight decay: {wd}\n')
            f.write(f'Clip value: {clipping_value}\n')

       
        os.rename(f'./generated/class_{self.target_class}',
                  f'./generated/class_{self.target_class}_blurfreq_{blur_freq}_blurrad_{blur_rad}_wd{wd}')
        return self.processed_image


def preprocess_and_blur_image(pil_im, resize_im=True, blur_rad=None):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print(
                "could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im.thumbnail((224, 224))

    #add gaussin blur to image
    if blur_rad:
        pil_im = pil_im.filter(ImageFilter.GaussianBlur(blur_rad))

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    im_as_ten = torch.from_numpy(im_as_arr).float()

    im_as_ten.unsqueeze_(0)
    
    if use_cuda:
        im_as_var = Variable(im_as_ten.cuda(), requires_grad=True)
    else:
        im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var
    