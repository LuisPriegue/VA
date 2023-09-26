import skimage
import numpy as np
from skimage import data

camera = data.camera()

normalized_camera = (camera-np.min(camera))/(np.max(camera)-np.min(camera))

print(normalized_camera)