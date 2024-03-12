import src.processing as processing
import src.utils as utils
import src.segmentation as segmentation
import src.plotter3d as plotter

if __name__ == "__main__":

    for image_index in range(23, 63):

        # Load and display the input image
        img = utils.load_img(filename=f"dataset/MRI/Axial/{image_index}.png")

        # Pre-process the image
        adjusted_image = processing.adjust_image(src=img)

        # Remove skull and obtain the brain and skull images
        brain, skull = processing.remove_skull(src=adjusted_image)

        # Adjust the brain
        adjusted_brain = processing.adjust_brain(src1=adjusted_image, src2=brain, src3=skull)

        # Perform segmentation using K-means
        images = segmentation.kmeans_segmentation(src1=adjusted_image, src2=adjusted_brain)

        # Display images
        preview = [adjusted_image, images["merged_no_skull"]]
        preview.extend([segment["colored"] for segment in images["segments"]])

        utils.show_img(preview, image_index)
        utils.save_img(src=images["merged_no_skull"], filename=f"dataset/results/merged_no_skull/{image_index}.png", overwrite=True)
    
    plotter.create_3d_image(src="dataset/results/results/merged_no_skull", dst="segmented_brain_colored", color=True, img_scale=4)