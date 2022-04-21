import matplotlib.pyplot as plt
import numpy as np


def show_vector_img(img_vector, img_dim):
    """ show numpy image-vector"""
    img_vector = np.reshape(img_vector, img_dim)
    plt.imshow(img_vector, cmap='gray', vmin=0, vmax=255)
    plt.show()
