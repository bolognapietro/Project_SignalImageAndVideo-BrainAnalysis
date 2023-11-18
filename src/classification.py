"""
Provides functions for the classification of segments of MRI brain images.
The goal is to classify the provided regions (segments) in:

- Grey matter.
- Cerebrospinal fluid.
- White matter.
"""

import cv2

def area_classification(images: dict) -> dict:
    """
    Classifies each segment, regardless of the algorithm used.
    This method is based on the area, specifically the ratio of white pixels to black pixels in each segment. 
    This allows for the classification of each segment; the cerebrospinal fluid has the smallest amount of white pixels, 
    grey matter has a moderate amount, and white matter has the highest amount of white pixels.

    :param images: Dictionary object resulting from segmentation algorithm.
    :type images: dict

    :return: A dictionary is provided, containing the segmented image where each segment is 
        now classified. The possible classes are:

        - Grey matter.
        - Cerebrospinal fluid.
        - White matter.
    :rtype: dict
    """

    # The order of these labels is based on the amount of pixels that each segment should have.
    # In fact, for example, the celebrospinal fluid segment should have the smaller amount of pixels.
    labels = ["Cerebrospinal fluid", "Grey matter", "White matter"]

    for segment in images["segments"]:

        black_and_white = segment["black_and_white"]

        # Convert the segment image to gray scale
        gray = cv2.cvtColor(src=black_and_white, code=cv2.COLOR_BGR2GRAY)

        # Threshold the image
        _, thresh = cv2.threshold(src=gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

        # Area will be used for classification
        segment["area"] = cv2.countNonZero(src=thresh)
    
    images["segments"].sort(key=lambda segment: segment["area"])
    
    for index, segment in enumerate(images["segments"]):

        # Assign classification label
        segment["label"] = labels[index]

        # Remove temp key
        del segment["area"]
    
    return images