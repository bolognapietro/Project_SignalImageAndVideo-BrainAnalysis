"""
Provides functions for the 2D segmentation of MRI brain images. 
It can be tested with different algorithms:

- `K-means <https://it.wikipedia.org/wiki/K-means>`_

The goal is to extract three regions (segments) of interest:

- Grey matter.
- Cerebrospinal fluid.
- White matter.

They can be later classified using the appropriate function inside the classify module.
"""

import cv2
import numpy as np

import distinctipy

def kmeans_segmentation(src: np.ndarray, k: int = 3) -> dict:
    """
    Executes a 2D segmentation of an MRI brain image using K-means algorithm.

    :param src: Image to be segmented.
    :type src: np.ndarray

    :param k: Optional, number of clusters used in K-means algorithm. Defaults to 3.
    :type k: int

    :return: A dictionary is provided, containing the segmented image where each segment is 
        represented by a distinct color. Additionally, the dictionary includes all segments 
        resulting from the segmentation process, presented both in color and in black and white.
    :rtype: dict
    """
    
    # Copy the original image
    img = src.copy()
    
    #img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Reshape the image to a single array of pixels
    pixels = img.reshape((-1, 3))

    # Convert the data type to float32
    pixels = np.float32(pixels)

    # Define criteria for kmeans (type, max_iter, epsilon)
    # type:
    #   - TERM_CRITERIA_EPS = Stop the algorithm iteration if the specified accuracy (epsilon) is achieved.
    #   - TERM_CRITERIA_MAX_ITER = Stop the algorithm after the specified number of iterations (max_iter).
    #   - TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER = Stop the algorithm when one of the criteria is satysfied.
    # max_iter: Maximum number of iterations.
    # epsilon: Required accuracy.

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(data=pixels, K=k, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Map the labels to their respective centers
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(img.shape)

    # Retrieve the k colors
    segment_colors = list(set(segmented_image.ravel().tolist()))

    # The final resut will be a dict where:
    # - completed will be the final image where each segmentation will have a different color
    # - segments will be an array of all the extracted segments. Each of them will have a colored and black and white version

    images = {
        "complete": None, 
        "segments": []
    }

    # Generate k random distinct color
    colors = []

    for color in distinctipy.get_colors(n_colors=len(segment_colors)):
        color = [int(channel*255) for channel in color]
        colors.append(color)

    for index, segment_color in enumerate(segment_colors):

        segment_color = (segment_color, segment_color, segment_color)

        # Create a copy of the segmented image so i can highlight a specific segment
        segment_image = segmented_image.copy()

        # Create the mask
        mask = cv2.inRange(src=segment_image, lowerb=segment_color, upperb=segment_color)
        
        # Apply the mask
        segment_image[mask <= 0] = [0,0,0]

        # Generate a black and white version and a colored one
        segment = {
            "colored": None,
            "black_and_white": None
        }

        # Colored version
        colored = segment_image.copy()
        colored[mask > 0] = colors[index]
        segment["colored"] = colored

        # Black and white version
        black_and_white = segment_image.copy()
        black_and_white[mask > 0] = (255,255,255)
        segment["black_and_white"] = black_and_white

        images["segments"].append(segment)

    # Fix the segments

    # Convert the original image to gray scale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Find contours
    # The background is inside the segment that also contains some colors of the brain associated to the same class
    # For this reason, we first find the contour of the brain and then exclude the external part (background) from the internal one
    
    contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(a=gray)

    # Draw the contour on the mask
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255), thickness=cv2.FILLED)

    final_image = None
    fixed_segments = []

    for segment in images["segments"]:

        colored = segment["colored"]
        black_and_white = segment["black_and_white"]

        # Use the mask to keep only the region inside the contour in the original imageù
        colored = cv2.bitwise_and(src1=colored, src2=colored, mask=mask)
        black_and_white = cv2.bitwise_and(src1=black_and_white, src2=black_and_white, mask=mask)

        fixed_segments.append({
            "colored": colored,
            "black_and_white": black_and_white
        })

        # Add the highlighted area to the final image
        if final_image is None:
            final_image = colored
        
        else:
            # Keep the colored pixel of final_image if final_image is different from 0 and colored not
            # Keep the colored pixel of colored if colored is different from 0 and final_image not
            # Keep the colored pixel of final_image if both final_image and colored are different from 0
            mask1 = final_image > 0
            mask2 = colored > 0

            final_image = np.where(mask1, final_image, np.where(mask2, colored, 0))
    
    images["complete"] = final_image
    images["segments"] = fixed_segments

    return images