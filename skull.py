import cv2
import numpy as np

import src.utils as utils
import src.processing as processing
import src.segmentation as segmentation

for i in range(19,133):

    print(i)

    # Test over all the images (Axial)
    img = utils.load_img(f"img/MRI/Axial/{i}.png")
    original_image = img.copy()

    # Adjust image
    img = processing.adjust_image(img)

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Threshold the image so we can separate the skull from the brain
    _, thresh = cv2.threshold(src=gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

    # Extract all contours
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    # Get the two largest contours based on their areas
    contours = [[cv2.contourArea(contour), contour] for contour in contours]
    contours.sort(key=lambda item: item[0])
    contours.reverse()
    contours = contours[:2]
    
    # If we manage to extract two contours (internal line and external line of the skull)
    if len(contours) == 2:
        
        # Get the internal contour of the skull
        contours = [contour[1] for contour in contours]

        # Erase what's out of it
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contours[1]], 0, (255), thickness=cv2.FILLED)

        brain = img.copy()
        brain[mask == False] = (0,0,0)

        # Adjust border: erasing it for n times
        for _ in range(8):
            gray = cv2.cvtColor(src=brain, code=cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image=brain, contours=contours, contourIdx=-1, color=(0,0,0), thickness=1)

        # Now restore the original image but only where there are pixels of the tmp image (the processed one)
        gray = cv2.cvtColor(src=brain, code=cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)

        contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255), thickness=cv2.FILLED)

        brain = original_image.copy()
        brain[mask == False] = (0,0,0)

    brain = processing.adjust_image(brain)
    gray = cv2.cvtColor(src=brain, code=cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(src=gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)
    
    utils.show_img(brain)

    '''images = segmentation.kmeans_segmentation(tmp)

    grid = [original_image, tmp]
    grid.append(images["complete"])
    grid.extend([image["colored"] for image in images["segments"]])

    utils.show_img(grid)'''

    #utils.show_img([original_image, tmp],ncolumns=2)