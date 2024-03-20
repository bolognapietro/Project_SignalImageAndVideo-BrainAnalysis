"""
Provides basic functions related to images.
"""

import numpy as np
import cv2

from os.path import isfile

from typing import Union

def show_img(src: Union[np.ndarray, list], winname: str = "Image", ncolumns: int = 3) -> None:
    """
    Shows an image or an array of images.

    :param src: Image(s) to be displayed.
    :type src: np.ndarray or list[np.ndarray]

    :param winname: Optional, window name. Defaults to "Image".
    :type winname: str

    :param ncolumns: Optional, number of columns if multiple images are displayed. 
        It's ignored if src param is a single image. Defaults to 3.
    :type ncolumns: int

    :return: None
    :rtype: None
    """

    # Check if the parameter type is correct
    assert type(src) in [np.ndarray, list], "Invalid source type."

    # Extract the image if the list is composed by one image only
    if type(src) is list and len(src) == 1:
        src = src[0]

    # If src is a list, create the grid of images
    if type(src) is list:

        assert len(src), "Empty array."
        
        # Get largest image
        max_image = max(src, key=lambda image: image.shape)

        columns = []
        rows = []

        for image in src:
            
            # Since all the image of the grid must have the same size, if an image has a smaller size,
            # we use the largest image as a black background and put it in the center
            if image.shape != max_image.shape:
                
                # Create black background
                background = np.zeros_like(a=max_image)

                # Get the dimensions of the background image
                background_height, background_width = background.shape[:2]

                # Get the dimensions of the center image
                center_height, center_width = image.shape[:2]

                # Calculate the coordinates to place the center image in the center of the background
                start_x = (background_width - center_width) // 2
                start_y = (background_height - center_height) // 2

                end_x = start_x + center_width
                end_y = start_y + center_height

                # Copy the color channels from the center image to the specified region in the background
                background[start_y:end_y, start_x:end_x] = image

                image = background

            # Create the column
            columns.append(image)
            
            if len(columns) == ncolumns:
                # Stack the images
                image = np.hstack(tup=columns)

                # Append the row of images to the rows list
                rows.append(image)

                columns = []

        # Do the same if there is an incompleted column
        if len(columns):
            
            # In this case i have to fill the missing cells with black images
            for _ in range(ncolumns - len(columns)):
                columns.append(np.zeros_like(a=columns[0]))

            image = np.hstack(tup=columns)
            rows.append(image)

        # Stack the rows
        src = np.vstack(tup=rows) 

    # Display the image(s)
    scale_percent = 70 # percent o.f original size
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(src, dim)

    cv2.imshow(winname=str(winname), mat=resized)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

def load_img(filename: str) -> np.ndarray:
    """
    Loads an image.

    :param filename: Source file.
    :type filename: str

    :return: Loaded image.
    :rtype: np.ndarray
    """

    # Check if file exists
    assert isfile(path=filename), f'"{filename}" doesn\'t exist.'

    # Load the image
    return cv2.imread(filename=filename)

def save_img(src: np.ndarray, filename: str, overwrite: bool = True) -> bool:
    """
    Saves an image.

    :param src: Image to be saved.
    :type src: np.ndarray

    :param filename: Destination file.
    :type filename: str

    :param overwrite: Optional, if true, the specified destination file will be overwritten if it exists. 
        Defaults to true.
    :type overwrite: bool

    :return: true on success, false otherwise. If the destination file exists and overwrite is set on false,
        the function will return false.
    :rtype: bool
    """

    try:

        if not isfile(path=filename) or (isfile(path=filename) and overwrite):
            return cv2.imwrite(filename=filename, img=src)

    except:
        pass
    
    return False
