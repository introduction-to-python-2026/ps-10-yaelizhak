from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection
image = load_image('.tests/lena.jpg')  
clean_image = median(image, ball(3))  
edgeMAG = edge_detection(clean_image)
edge_binary = edgeMAG > 50
plt.imshow(edge_binary, cmap='gray')
plt.axis('off')
plt.show()
