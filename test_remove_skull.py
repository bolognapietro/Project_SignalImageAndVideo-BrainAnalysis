import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import extrema
from skimage.segmentation import watershed as skwater

def ShowImage(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_skull(image):
    #Read in image
    img = cv2.imread(image)
    #Convert the image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # ret: threshold value, thresh: binary image 
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

    #Make a histogram of the intensities in the grayscale image
    #plt.hist(gray.ravel(),256)
    #plt.show()

    #Threshold the image to binary using Otsu's method
    colormask = np.zeros(img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))
    blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component

    removed_skull = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    removed_skull[brain_mask==False] = (0,0,0)

    return gray, thresh, blended, removed_skull



image = "img/MRI/Axial/17.png"
gray, thresh, blended, removed_skull = remove_skull(image)

# Convert gray to a 3-channel image
gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Concatenate images horizontally
combined_image = np.hstack((gray_colored, thresh_colored, blended, removed_skull))

# Display the concatenated image
ShowImage('Gray Image and Skull Removed', combined_image)