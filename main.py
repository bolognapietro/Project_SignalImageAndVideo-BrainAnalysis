import src.processing as processing
import src.utils as utils
import src.segmentation as segmentation

import cv2
import numpy as np

if __name__ == "__main__":

    for image_index in range(23, 63):

        # Load and display the input image
        img = utils.load_img(filename=f"img/MRI/Axial/{image_index}.png")
        #utils.show_img(src=img)

        # Pre-process image
        img1 = processing.adjust_image(src=img)
        #utils.show_img(src=img1)

        # Remove skull
        img2 = processing.remove_skull(src=img1)

        # Find closer contour to the brain
        img3, mask = processing.find_best_brain_contour(original_image=img1, brain=img2)

        # Segmentation
        images = segmentation.kmeans_segmentation(src=img3)

        # Adjust each image by removing the contour of the skull (previously added)
        grid = [img]

        images["complete"][mask != False] = (0, 0, 0)
        grid.append(images["complete"])

        for image in images["segments"]:
            image["black_and_white"][mask != False] = (0, 0, 0)
            image["colored"][mask != False] = (0, 0, 0)

            grid.append(image["colored"])

        utils.show_img(grid)
