from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt

clean_image = median(Timon, ball(2))
edgeMAG = edge_detection(clean_image)
plt.imshow(edgeMAG, cmap = 'gray')

plt.hist(edgeMAG.ravel(), bins=50)
plt.xlabel('Edge magnitude value')
plt.ylabel('Frequency')
plt.show()

edgeMAG_copy = edgeMAG.copy()
edgeMAG_copy[edgeMAG_copy < 50] = 0
edgeMAG_copy[edgeMAG_copy >= 50] = 1
plt.imshow(edgeMAG_copy, cmap = 'gray')

from PIL import Image
edge_image = Image.fromarray((edgeMAG_copy * 255).astype(np.uint8))
edge_image.save('my_edges(1).png')

