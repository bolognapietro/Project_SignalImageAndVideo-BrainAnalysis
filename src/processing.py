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

def remove_skull(src: np.ndarray) -> np.ndarray:
    """
    Removes the skull from an MRI brain image.

    :param src: Image to be processed.
    :type src: np.ndarray.

    :return: processed image.
    :rtype: np.ndarray
    """

    # Copy the image
    img = src.copy()

    #? Remove the skull

    # Convert to grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(src=gray, thresh=95, maxval=255, type=cv2.THRESH_BINARY)

    # Get each area of pixels of the image (label 0 is background)
    _, markers = cv2.connectedComponents(image=thresh)
    marker_area = [np.sum(a=markers == m) for m in range(np.max(a=markers)) if m != 0] 

    # <= 1 because sometimes we work with brain images that doesn't have the skull
    #TODO fix this if we have multiple small marker area but non the skull
    if len(marker_area) <= 1:
        return img

    # Get largest area (excluding background)
    largest_component = np.argmax(a=marker_area)+1

    # Get pixels of the extracted area
    brain_mask = markers == largest_component

    # Color of black all the pixels that aren't inside of the specified area
    brain = img.copy()
    brain[brain_mask == False] = (0,0,0)

    # Color of black all the pixels that are inside of the specified area
    skull = img.copy()
    skull[brain_mask == True] = (0,0,0)

    #? Adjust the extracted brain / removed skull

    # Convert to grayscale
    gray = cv2.cvtColor(src=skull, code=cv2.COLOR_BGR2GRAY)

    # Blur the skull so we can remove pixels inside it (they belong to the brain)
    #blurred = cv2.GaussianBlur(src=gray, ksize=(21, 21), sigmaX=0)

    # Threshold the image
    _, thresh = cv2.threshold(src=gray, thresh=95, maxval=255, type=cv2.THRESH_BINARY)

    fixed_skull = cv2.cvtColor(src=thresh, code=cv2.COLOR_GRAY2BGR)

    # Now we have three images:
    # - (brain) the extracted brain
    # - (skull) the skull with pixels inside that belongs to the brain
    # - (fixed_skull) the fixed skull without pixels of the brain
    # We have to do this operation:  brain + (skull - fixed_skull)

    # skull - fixed_skull = brain_pixels
    mask = cv2.cvtColor(src=fixed_skull, code=cv2.COLOR_BGR2GRAY)

    brain_pixels = skull.copy()
    brain_pixels[mask > 0] = (0,0,0) # Replace all the pixels of brain_pixels corresponding to fixed_skull with 0 (black)

    # brain + brain_pixels
    brain_pixels_gray = cv2.cvtColor(src=brain_pixels, code=cv2.COLOR_BGR2GRAY)
    brain_pixels_colored = brain_pixels_gray > 0 # Get non-black pixels of brain_pixels

    brain[brain_pixels_colored] = brain_pixels[brain_pixels_colored] # Add the retrieved pixels to the brain image

    #? Adjust border

    # Convert the image to grayscale
    gray = cv2.cvtColor(src=brain, code=cv2.COLOR_BGR2GRAY)

    # Blur the image so we can have soft margins
    gray = cv2.GaussianBlur(src=gray, ksize=(31, 31), sigmaX=0)

    # Threshold the image
    # thresh helps if something doesn't work
    _, thresh = cv2.threshold(src=gray, thresh=80, maxval=255, type=cv2.THRESH_BINARY)

    # Apply the mask
    brain[thresh <= 0] = (0,0,0)

    #? Finally fill any hole inside the brain

    gray = cv2.cvtColor(src=brain, code=cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(image=gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with zeros (black)
    mask = np.zeros_like(a=gray)

    # Draw the contour on the mask
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(255), thickness=cv2.FILLED)

    # Set all pixels outside the contour to 0
    brain = img.copy()
    brain = cv2.bitwise_and(src1=brain, src2=brain, mask=mask)

    # Return the image
    return brain