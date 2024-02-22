"""
Provides advanced functions for processing MRI brain images.
"""

import numpy as np
import cv2
from skimage.restoration import denoise_tv_bregman
from skimage.util import img_as_ubyte

def adjust_image(src: np.ndarray) -> np.ndarray:
    """
    Processes an MRI image by performing the following operations:
    
    - Denoising the image using split-Bregman optimization.
    - Histogram equalization.
    - Setting the background color to black.

    :param src: Image to be adjusted.
    :type src: np.ndarray

    :return: Adjusted image.
    :rtype: np.ndarray
    """

    # Copy the image
    img = src.copy()

    #? Convert to grayscale

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    #? Denoise the image using split-Bregman optimization

    denoised = denoise_tv_bregman(gray, 4)
    denoised = img_as_ubyte(denoised)

    #? Perform histogram equalization only on the brain

    equalized = cv2.equalizeHist(denoised)
    equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # Extract the brain
    _, thresh = cv2.threshold(src=denoised, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Create a black image (all zeros) with the same size as img
    merged = np.zeros_like(img)

    # Draw the contour of the brain on merged
    cv2.drawContours(merged, [contours[0]], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Create a mask for the region inside the contour
    mask = np.zeros_like(denoised)
    cv2.drawContours(mask, [contours[0]], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Assign equalized inside the brain
    merged[mask == 255] = equalized[mask == 255]

    # Assign denoised outside the brain. 
    # In this way we can denoise the background and make it all black, without doing equalization on it
    denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    merged[mask == 0] = denoised[mask == 0]

    return merged

def merge_images(src1: np.ndarray, src2: np.ndarray):
    """
    Merges two images.

    :param src1: Image1.
    :type src: np.ndarray.

    :param src2: Image2.
    :type src: np.ndarray.

    :return: Merged image.
    :rtype: np.ndarray
    """

    gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Create the mask
    mask_inv = cv2.bitwise_not(thresh)

    # Use the mask to color non-black pixels in src1 with corresponding pixels from src2
    result = cv2.bitwise_and(src1, src1, mask=mask_inv)
    result = cv2.add(result, src2)

    return result

def remove_skull(src: np.ndarray) -> np.ndarray:
    """
    Removes the skull from an MRI brain image.

    :param src: Image to be processed.
    :type src: np.ndarray.

    :return: Processed image.
    :rtype: np.ndarray
    """

    # Copy the image
    img = src.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image using OTSU method. In this way it is possible to use the optimal threshold value based on the image content/histogram
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

    # Create a mask
    mask = np.zeros_like(img)

    # Add each area to the mask except for the first (skull) and the smaller ones
    for marker, area in sorted_markers:

        if marker == 1:
            continue

        if area < 4500:
            continue

        mask[markers == marker] = (255, 255, 255)

    # Apply the mask to the original image
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    brain = img.copy()
    brain[mask == False] = (0, 0, 0)

    mask = np.zeros_like(img)
    mask[markers == 1] = (255, 255, 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    skull = img.copy()
    skull[mask == False] = (0, 0, 0)

    return brain, skull

def find_best_brain_contour(src: np.ndarray) -> tuple:
    """
    Find the best contour of the brain.

    :param src: Brain image to be processed.
    :type src: np.ndarray.

    :return: Processed image and mask of the contour.
    :rtype: tuple
    """

    brain1 = src.copy()
    brain2 = src.copy()

    # Extract all the external point top/bottom approach and fill the columns of pixels of white
    for x in range(0, brain1.shape[1]):

        if np.all(brain1[:, x] == 0):
            continue
        
        non_zero = cv2.findNonZero(brain1[:, x])
    
        y_top = non_zero[0][0][1]
        y_bottom = non_zero[-1][0][1]

        brain1[y_top:y_bottom+1, x] = (255, 255, 255)

    # Extract all the external point left/right approach and fill the columns of pixels of white
    for y in range(0, brain1.shape[0]):

        if np.all(brain1[y, :] == 0):
            continue
        
        non_zero = cv2.findNonZero(brain1[y, :])

        x_left = non_zero[0][0][1]
        x_right = non_zero[-1][0][1]

        brain1[y, x_left:x_right+1] = (255, 255, 255)

    # Create and apply mask
    gray = cv2.cvtColor(brain1, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(brain1.shape, dtype=np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=1)

    brain2[mask != False] = (255, 255, 255)

    return brain2, mask