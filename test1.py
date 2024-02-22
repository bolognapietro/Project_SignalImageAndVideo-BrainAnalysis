import cv2
import numpy as np

import src.utils as utils
import src.processing as processing
import src.segmentation as segmentation

for i in range(21, 63):

    # Load image
    img = utils.load_img(f"img/MRI/Axial/{i}.png")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresh the image
    _, thresh = cv2.threshold(src=gray, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # Create mask (all black for now)
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[thresh != 0] = (0, 0, 255)

    # Extract areas inside the image
    _, markers = cv2.connectedComponents(thresh)

    # Calculate the size of each area
    marker_areas = [np.sum(markers == m) for m in range(np.max(markers)) if m != 0]

    # Sort each marker by area
    sorted_markers = sorted(enumerate(marker_areas, start=1), key=lambda x: x[1], reverse=True)

    # Apply the color map to the markers in the image
    colored_markers = np.zeros_like(img)

    for marker, area in sorted_markers:

        if marker == 1:
            continue

        if area < 4000:
            continue

        colored_markers[markers == marker] = (255, 255, 255)

    utils.show_img([img, colored_markers])

