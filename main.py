import src.processing as processing
import src.utils as utils
import src.segmentation as segmentation

if __name__ == "__main__":

    # Load and display the input image
    img = utils.load_img(filename=r"img/brain1.png")
    utils.show_img(src=img)

    # Pre-process image
    img1 = processing.adjust_image(src=img)
    utils.show_img(src=img1)

    # Remove skull
    #img2 = processing.remove_skull(src=img1)
    #utils.show_img(src=img2)

    # Segmentation
    images = segmentation.kmeans_segmentation(src=img1)

    grid = [img]
    grid.append(images["complete"])
    grid.extend([image["colored"] for image in images["segments"]])

    utils.show_img(grid)
