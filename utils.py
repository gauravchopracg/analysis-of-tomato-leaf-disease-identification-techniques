# contains helper functions for data preprocessing and analysis

import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models

def convert2grayscale(img_as_arr):
    """
        Converts 3d image to grayscale
    Args:
        img_as_arr (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_img (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_img = np.sum(np.abs(img_as_arr), axis=0)
    img_max = np.percentile(grayscale_img, 99)
    img_min = np.min(grayscale_img)
    grayscale_img = (np.clip((grayscale_img - img_min) / (img_max - img_min), 0, 1))
    grayscale_img = np.expand_dims(grayscale_img, axis=0)
    return grayscale_img

def save_gradients_images(gradients, file_name):
    """
        Exports the original gradients image
    Args:
        gradients (np arr): Numpy array of the gradients with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Normalize
    gradients = gradients - gradients.min()
    gradients /= gradients.max()
    # Save image
    path_to_file = os.path.join('results', file_name + '.jpg')
    save_image(gradients, path_to_file)

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('results', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # Save grayscale heatmap
    path_to_file = os.path.join('results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)

def apply_colormap_on_image(org_img, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def format_np_output(np_arr):
    """
        Converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        img_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

def save_image(img, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        img_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(img, (np.ndarray, np.generic)):
        img = format_np_output(img)
        img = Image.fromarray(img)
    img.save(path)

def preprocess_image(pil_img, resize_img=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_img (bool): Resize to 224 or not
    returns:
        img_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_img:
        pil_img.thumbnail((512, 512))
    img_as_arr = np.float32(pil_img)
    img_as_arr = img_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(img_as_arr):
        img_as_arr[channel] /= 255
        img_as_arr[channel] -= mean[channel]
        img_as_arr[channel] /= std[channel]
    # Convert to float tensor
    img_as_ten = torch.from_numpy(img_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    img_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    img_as_var = Variable(img_as_ten, requires_grad=True)
    return img_as_var

def recreate_image(img_as_var):
    """
        Recreates images from a torch variable
    Args:
        img_as_var (torch variable): Image to recreate
    returns:
        recreated_img (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_img = copy.copy(img_as_var.data.numpy()[0])
    for c in range(3):
        recreated_img[c] /= reverse_std[c]
        recreated_img[c] -= reverse_mean[c]
    recreated_img[recreated_img > 1] = 1
    recreated_img[recreated_img < 0] = 0
    recreated_img = np.round(recreated_img * 255)

    recreated_img = np.uint8(recreated_img).transpose(1, 2, 0)
    return recreated_img

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.
    Args:
        example_index (int): Image id to use from examples
    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('Tomato___Bacterial_spot.jpg', 56),
                    ('Tomato___Septoria_leaf_spot.jpg', 243),
                    ('Tomato___Tomato_mosaic_virus.jpg', 72))
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Load the trained model
    pretrained_model = torch.load(‘tomato.pth’)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)
