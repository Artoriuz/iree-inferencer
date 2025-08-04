import numpy as np


def unstack(image):
    image = np.squeeze(image)
    x, y, z = np.unstack(image, axis=-1)
    return (x, y, z)


def stack(x, y, z):
    x = np.squeeze(x)
    y = np.squeeze(y)
    z = np.squeeze(z)
    image = np.stack((x, y, z), axis=-1)
    return image
