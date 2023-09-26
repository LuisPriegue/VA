import os
import skimage
import numpy as np
from skimage import io

filename = os.path.join(skimage.data_dir, 'moon.png')
moon = io.imread(filename)
normalized_moon = (moon-np.min(moon))/(np.max(moon)-np.min(moon))
print(normalized_moon)