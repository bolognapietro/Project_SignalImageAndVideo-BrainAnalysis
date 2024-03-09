"""
Provides functions for the 2D segmentation of MRI brain images. 
It can be tested with different algorithms:

- `K-means <https://it.wikipedia.org/wiki/K-means>`_

The goal is to extract three regions (segments) of interest:

- Grey matter.
- Cerebrospinal fluid.
- White matter.

They can be later classified using the appropriate function inside the classification module.
"""

import cv2
import numpy as np

import src.classification as classification
import src.processing as processing

def kmeans_segmentation(src1: np.ndarray, src2: np.ndarray, k: int = 3) -> dict:
    """
    Executes a 2D segmentation of an MRI brain image using K-means algorithm.

    :param src1: Original image.
    :type src1: np.ndarray

    :param src2: Image to be segmented. It should be the brain.
    :type src2: np.ndarray

    :param k: Optional, number of clusters used in K-means algorithm. Defaults to 3.
    :type k: int

    :return: A dictionary is provided, containing the segmented image where each segment is 
        represented by a distinct color (see classification module for more details). Additionally, the dictionary also includes
        the merged image.
    :rtype: dict
    """
    
    # Copy the original image
    original_image = src1.copy()
    img = src2.copy()

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

    # Generate a first version of the segments
    segments = []

    for segment_color in segment_colors:

        segment_color = (segment_color, segment_color, segment_color)

        # Create a copy of the segmented image so i can highlight a specific segment
        segment_image = segmented_image.copy()

        # Create the mask
        mask = cv2.inRange(src=segment_image, lowerb=segment_color, upperb=segment_color)
        
        # Apply the mask
        segment_image[mask <= 0] = [0,0,0]

        segment = segment_image.copy()
        segment[mask > 0] = (255,255,255)
        segments.append(segment)

    # Adjust the segments

    # Convert the original image to gray scale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(a=gray)

    # Draw the contour on the mask
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=cv2.FILLED)

    fixed_segments = []

    for segment in segments:

        # Apply the mask to keep the segmentation inside the brain
        segment[mask == False] = (0, 0, 0)

        fixed_segments.append(segment)

    # Classify segments:
    classified_segments = classification.segments_classification(segments=segments)

    # Create merged image
    merged_image = np.zeros_like(img)

    for segment in classified_segments:
        merged_image = processing.merge_images(src1=merged_image, src2=segment["colored"])

    # Adjust the cerebrospinal fluid
    for segment in classified_segments:

        if segment["label"] != classification.LABELS.CEREBROSPINAL_FLUID:
            continue
        
        # Debug
        #gray = cv2.cvtColor(src=segment["colored"], code=cv2.COLOR_BGR2GRAY)
        #contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        #contours = [contour for contour in contours if cv2.contourArea(contour) >= 700]
        #cv2.drawContours(image=segment["colored"], contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2)

        gray = cv2.cvtColor(src=merged_image, code=cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image=segment["colored"], contours=contours, contourIdx=-1, color=segment["color"], thickness=2)
        cv2.drawContours(image=segment["segment"], contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2)

        break

    # Adjust merged image
    merged_image = processing.merge_images(src1=merged_image, src2=[segment["colored"] for segment in classified_segments if segment["label"] == classification.LABELS.CEREBROSPINAL_FLUID][0])

    #TODO Adjust internal parts of the brain

    return {
        "merged_no_skull": merged_image,
        "merged_skull": processing.merge_images(src1=original_image, src2=merged_image),
        "segments": classified_segments
    }