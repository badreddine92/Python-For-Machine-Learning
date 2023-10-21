import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#Simplifying the image's numpy table
image = plt.imread('bacteria.png')
image = image[:, :, 0]
plt.imshow(image, cmap='gray')
plt.show()

#Histogramme shows where are the darker pixels
image_2 = np.copy(image)
plt.hist(image_2.ravel(), bins=255)
plt.show()

#Removing backroung from the image by letting only darker pixels
image = image < 0.6
plt.imshow(image)
plt.show()

#Cleaning the background using ndimage
open_x = ndimage.binary_opening(image)
plt.imshow(open_x)
plt.show()

#Counting the number of bacteria in the image and showing image label
image_label, n_labels = ndimage.label(open_x)
print(f'{n_labels} is the number of bacteria in our image')

plt.imshow(image_label)
plt.show()

#Counting the number of pixels for each bacteria 
sizes = ndimage.sum(image, image_label, range(n_labels))

#Creating a small data set with scatter 
plt.scatter(range(n_labels), sizes)
plt.show()
