from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt

image = load_image('Timon')  
clean_image = median(image, ball(3))  
edgeMAG = edge_detection(clean_image)
edge_binary = (edgeMAG > 50).astype(np.uint8) * 255
edge_image = Image.fromarray(edge_binary)
edge_image.save('my_edges(1).png')

