"""
Provides functions for the classification of segments of MRI brain images.
The goal is to classify the provided regions (segments) in:

- Grey matter.
- Cerebrospinal fluid.
- White matter.
"""

import cv2

from enum import Enum

class LABELS(Enum):
    CEREBROSPINAL_FLUID = "Cerebrospinal fluid"
    GREY_MATTER = "GREY_MATTER"
    WHITE_MATTER = "WHITE_MATTER"

def segments_classification(segments: list, color_segments: bool = True) -> dict:
    """
    Classifies each segment, regardless of the algorithm used.
    This method is based on the area, specifically the ratio of colored pixels to black pixels in each segment. 
    This allows for the classification of each segment; the cerebrospinal fluid has the smallest amount of white pixels, 
    grey matter has a moderate amount, and white matter has the highest amount of white pixels.

    :param images: List of segments obtained from segmentation algorithm.
    :type images: list

    :param color_segments: Optional, color classified segments. Defaults to true.
    :type color_segments: bool

    :return: A dict is provided, containing the labelled images where each segment is 
        now classified. The possible classes are:

        - Grey matter.
        - Cerebrospinal fluid.
        - White matter.
    :rtype: dict
    """

    # The order of these labels is based on the amount of pixels that each segment should have.
    # In fact, for example, the celebrospinal fluid segment should have the smaller amount of pixels.
    labels = [LABELS.CEREBROSPINAL_FLUID, LABELS.GREY_MATTER, LABELS.WHITE_MATTER]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # Colors in BGR format

    segment_with_area = []

    for segment in segments:

        # Convert the segment image to gray scale
        gray = cv2.cvtColor(src=segment, code=cv2.COLOR_BGR2GRAY)

        # Threshold the image
        _, thresh = cv2.threshold(src=gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

        # Area will be used for classification
        segment_with_area.append([segment, cv2.countNonZero(src=thresh)])
    
    # Sort segment based on area
    segment_with_area = sorted(segment_with_area, key=lambda segment: segment[1])

    classified_segments = []

    for index, segment in enumerate(segment_with_area):

        label = labels[index]
        color = colors[index]
        segment = segment[0].copy()

        colored = None

        # If color option is enabled, color the segment
        if color_segments:
            mask = cv2.inRange(segment, (255, 255, 255), (255, 255, 255))

            colored = segment.copy()
            colored[mask == 255] = color

        classified_segments.append({
            "label": label,
            "color": color,
            "segment": segment,
            "colored": colored
        })

    return classified_segments