import src.processing as processing
import src.utils as utils
import src.segmentation as segmentation

if __name__ == "__main__":

    for image_index in range(23, 63):

        # Load and display the input image
        img = utils.load_img(filename=f"img/MRI/Axial/{image_index}.png")

        # Pre-process image
        img1 = processing.adjust_image(src=img)

        # Remove skull and obtain brain image
        brain, skull = processing.remove_skull(src=img1)

        # Adjust brain
        brain = processing.adjust_brain(src1=img1, src2=brain, src3=skull)

        # Segmentation
        images = segmentation.kmeans_segmentation(src1=img1, src2=brain)

        # Display images
        preview = [img1, images["merged_skull"]]
        preview.extend([segment["colored"] for segment in images["segments"]])

        utils.show_img(preview, image_index) # 50
