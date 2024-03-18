"""
Provides functions for 3D plotting.
"""

import os
import numpy as np
import cv2 as cv
import plotly.graph_objects as go
import src.utils as utils

def convert_to_png(src: np.ndarray) -> np.ndarray:
    """
    Converts an image to PNG format with an alpha channel.

    :param src: Input image.
    :type src: np.ndarray

    :return: Converted image in PNG format with an alpha channel.
    :rtype: np.ndarray
    """

    # Calculate the sum of the image along the last axis to obtain a binary mask where non-zero values indicate the presence of pixels.
    alpha = np.sum(src, axis=-1) > 0
    
    # Convert the binary mask to an 8-bit integer array by multiplying it with 255.
    alpha = alpha.astype(np.uint8) * 255
    
    # Stack the original image and the alpha channel together along the third axis to create an image with an alpha channel.
    result = np.dstack((src, alpha))
    
    return result

def interpolate(p_from: np.ndarray, p_to: np.ndarray, num: int) -> np.ndarray:
    """
    Interpolates between two points in 3D space.

    :param p_from: Starting point in 3D space.
    :type p_from: np.ndarray

    :param p_to: Ending point in 3D space.
    :type p_to: np.ndarray

    :param num: Number of points to interpolate between p_from and p_to.
    :type num: int

    :return: An array of interpolated points between p_from and p_to.
    :rtype: np.ndarray
    """

    # Calculate the direction vector
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)

    # Calculate the distance between each interpolated point
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    # Generate a list of interpolated points
    ret_vec = [p_from + direction * distance * i for i in range(num)]

    return np.array(ret_vec)


def plotImage(ply: go.Figure, img: np.ndarray, z_index: float, size: np.ndarray = np.array((1, 1)), img_scale: int = 2, color: bool = False) -> None:
    """
    Plot an image on a 3D plot.

    :param ply: Plotly figure object to add the image to.
    :type ply: plotly.graph_objects.Figure

    :param img: Image to plot.
    :type img: np.ndarray

    :param index: Z-index of the image.
    :type index: float

    :param size: Optional, size of the image in the plot. Defaults to np.array((1, 1)).
    :type size: np.ndarray

    :param img_scale: Optional, scale factor to resize the image. Defaults to 2.
    :type img_scale: int

    :param color: Optional, whether to plot the image in color. Defaults to False.
    :type color: bool

    :return: None
    :rtype: None
    """

    # Calculate the new size of the image and resize it
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    # Array representing the four corners of the image in 3D space.
    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    # Initialize 2D arrays with the same size as the image, to store the x, y, and z 
    # coordinates of each pixel in the image.
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    
    # Interpolate function is used to generate a series of interpolated points 
    # between the first and third corners of an image.
    lx = interpolate(corners[0], corners[2], img_size[0])

    # These points are then assigned to the corresponding positions in the xx, yy, and zz arrays.
    xx[:, 0] = lx[:, 0]
    yy[:, 0] = lx[:, 1]
    zz[:, 0] = lx[:, 2]

    # Interpolate between the second and fourth corners to get the left side of the image
    ly = interpolate(corners[1], corners[3], img_size[0])

    # Set the x, y, and z coordinates of the rightmost column of the image to the interpolated values
    xx[:, img_size[1] - 1] = ly[:, 0]
    yy[:, img_size[1] - 1] = ly[:, 1]
    zz[:, img_size[1] - 1] = ly[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    # Loop that iterates over each pixel in the image. 
    for i in range(img_size[0]):
        for j in range(img_size[1]):

            # If the alpha value of the pixel is 255 (fully opaque), it sets the corresponding position in the 'zz' array
            if img[i, j, 3] == 255:
                zz[i, j] = z_index
            
            # If the alpha value is not 255, it sets the corresponding position in the 'zz' array to None.
            else:
                zz[i, j] = None

    # If the 'color' flag is False, add a simple surface
    if not color:
        ply.add_surface(
            x=xx,
            y=yy,
            z=zz,
            showscale=False,
            opacity=1
        )
    
    # Otherwise, if the 'color' flag is True, add a trace with the correspondin color
    else:
        # Extract the RGB channels from the image and normalize the color values
        colors = (img[:, :, :3] / 255.0).reshape((-1, 3)) 
        colors_bgr = (colors * 255.0).reshape((-1, 3))[:, ::-1]

        ply.add_trace(
            go.Scatter3d(
                x=xx.flatten(),
                y=yy.flatten(),
                z=zz.flatten(),
                mode='markers',
                marker=dict(
                    size=1,
                    color=colors_bgr,
                    opacity=1
                ),
                showlegend=False
            )
        )

    return None

def create_3d_image(src: str, dst: str, color: bool = False, img_scale: int = 2) -> None:
    """
    Create a 3D image plot using a series of images.

    :param src: Directory path containing the input images.
    :type src: str

    :param dst: Name of the output HTML file.
    :type dst: str

    :param color: Optional, whether to plot the image in color. Defaults to False.
    :type color: bool

    :param img_scale: Optional, whether to plot the image in color. Defaults to False.
    :type color: intcla

    :return: None
    :rtype: None
    """

    ply = go.Figure()

    # List of files sorted in descending order.
    files = sorted(os.listdir(src), key=lambda x: int(x.split('.')[0]), reverse=True)

    z_index=0
    for filename in files:
        img = utils.load_img(os.path.join(src, filename))
        res = convert_to_png(src=img)
        
        plotImage(ply, res, z_index, size=np.array((1, img.shape[0] / img.shape[1])), img_scale=img_scale, color=color)
        z_index += 0.05


    ply.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1.5]),  # Set the limits of the x-axis.
            yaxis=dict(range=[0, 1.3]),  # Set the limits of the y-axis.
            zaxis=dict(range=[0, 15]),  # Set the limits of the z-axis.
        )
    )

    print("Loading the brain plot..")
    ply.show()
    
    print("Saving the plot...")
    ply.write_html(f"web/plot_html/{dst}.html")
