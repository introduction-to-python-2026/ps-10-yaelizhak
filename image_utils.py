from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    image = np.array(image)
    return image
    
def edge_detection(image): 
    gray_image = np.mean(image, axis = 2)
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    edgeX = np.convolve2d(gray_image, kernelX, mode = 'same', boundary = 'fill')
    edgeY = np.convolve2d(gray_image, kernelY, mode = 'same', boundary = 'fill')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
