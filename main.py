import src.processing as processing
import src.utils as utils
import src.segmentation as segmentation

if __name__ == "__main__":

    for image_index in range(23, 63):

        # Load and display the input image
        img = utils.load_img(filename=f"img/MRI/Axial/{image_index}.png")
        #utils.show_img(src=img)

        # Pre-process image
        img1 = processing.adjust_image(src=img)
        #utils.show_img(src=img1)

        # Remove skull
        brain, skull = processing.remove_skull(src=img1)

        # Find closer contour to the brain
        brain, mask = processing.find_best_brain_contour(src=brain)

        # Segmentation
        images = segmentation.kmeans_segmentation(src=brain)

        # Adjust each image by removing the contour of the skull (previously added)
        grid = [img]

        images["complete"][mask != False] = (0, 0, 0)
        grid.append(images["complete"])

        for image in images["segments"]:
            image["black_and_white"][mask != False] = (0, 0, 0)
            image["colored"][mask != False] = (0, 0, 0)

            grid.append(image["colored"])

        # Merge everything
        final_image = processing.merge_images(src1=img1, src2=images["complete"])

        # Display the result
        utils.show_img(final_image)

