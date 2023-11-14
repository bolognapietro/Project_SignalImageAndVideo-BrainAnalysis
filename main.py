import src.image_processing as image_processing
import src.utils as utils
import src.segmentation_2D as segmentation_2D

import cv2 
import numpy as np

if __name__ == "__main__":

    # Load and display the input image
    img = utils.load_img(filename=r"")
    utils.show_img(src=img)

    # Pre-process image
    img1 = image_processing.adjust_image(src=img)
    utils.show_img(src=img1)

    # Remove skull
    img2 = image_processing.remove_skull(src=img1)
    utils.show_img(src=img2)

    # Segmentation
    images = segmentation_2D.kmeans_segmentation(src=img2)

    grid = [img]
    grid.append(images["complete"])
    grid.extend([image["colored"] for image in images["segments"]])

    utils.show_img(grid)
