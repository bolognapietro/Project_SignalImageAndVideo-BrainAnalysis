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

        # Draw external contour of the skull for the segmentation
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(src=gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask since we will also use it later
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=1)

        img2[mask != False] = (255, 255, 255)

        # Segmentation
        images = segmentation.kmeans_segmentation(src=img2)

        # Adjust each image by removing the contour of the skull (previously added)
        grid = [img]

        images["complete"][mask != False] = (0, 0, 0)
        grid.append(images["complete"])

        for image in images["segments"]:
            image["black_and_white"][mask != False] = (0, 0, 0)
            image["colored"][mask != False] = (0, 0, 0)

            grid.append(image["colored"])

        utils.show_img(grid)
