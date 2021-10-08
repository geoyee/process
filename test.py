import numpy as np
from PIL import Image


png_path = "gt/BDCI2020_T000202.png"
png = np.asarray(Image.open(png_path))
print(np.unique(png))