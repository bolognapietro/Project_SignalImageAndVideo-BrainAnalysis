# Segmentation of MRI brain images

<img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>  <img src="https://img.shields.io/badge/html5-%23E34F26.svg?style=flat&logo=html5&logoColor=white" alt="HTML"/><img src="https://img.shields.io/badge/javascript-%23323330.svg?style=flat&logo=javascript&logoColor=%23F7DF1E" alt="JavaScript"/><img src="https://img.shields.io/badge/css3-%231572B6.svg?style=flat&logo=css3&logoColor=white" alt="CSS"/>

# Table of contents

-   [Introduction](#introduction)
-   [Code overview](#code-overview)
-   [Getting started](#getting-started)
-   [Contacts](#contacts)

# Introduction

The goal of this project is to develop an algorithm capable of performing the segmentation of a MRI brain dataset. The script specifically analyzes MRI images of the brain in axial orientation,  where the slices are acquired horizontally, like taking pictures from top to bottom. Finally, the analyzed images are merged to create a 3D representation of the segmentation.

#### Expected results

<img src="assets/demo/expected-results.png"/>

#### Obtained results

<img src="assets/demo/obtained-results.png"/>

#### 3D Segmentation

<p float="left" align="center">
    <img width="32%" src="assets/3d/s1.png"/>
    <img width="32%" src="assets/3d/s2.png"/>
    <img width="32%" src="assets/3d/s3.png"/>
</p>

#### Disclaimer

Aside from the segmentation aspect, the algorithm employed in this project is designed to generalize well even when applied to images not included in the dataset. However, its flexibility is somewhat limited compared to algorithms that utilize more advanced techniques, such as machine learning algorithms. Nonetheless, it can closely resemble real medical segmentations.

# Code overview

This project enables the processing of MRI brain images in various ways, all with the ultimate goal of segmentation. In theory, each brain image should undergo the following process. However, it's worth noting that this sequence is not strictly mandatory. Depending on the characteristics of certain images, some steps may be omitted:

1. **Image adjustment**: in this phase, the initial brain image undergoes adjustment, including denoising, histogram equalization, and setting the background to black. This phase is particularly important for segmentation, as it helps to equalize the image colors.

   ```python
   # Pre-process the image
   adjusted_image = processing.adjust_image(src=img)
   ```

2. **Skull removal**: In this phase, if present, the skull is removed, as segmentation focuses solely on the brain. This is achieved by analyzing each area of the images and retaining the larger ones (above a certain threshold), excluding the largest one, which theoretically corresponds to the skull.

   ```python
   # Remove skull and obtain the brain and skull images
   brain, skull = processing.remove_skull(src=img)
   ```

3. **Brain adjustment**: in this phase, all objects remaining in the image are examined to determine if they could be part of the brain. The analysis involves subtracting both the skull and the brain from the original image. Any objects intersecting (partially or completely) with the brain are considered as potential components of it.

   ```python
   # Adjust the brain
   brain = processing.adjust_brain(src1=img, src2=brain, src3=skull)
   ```

4. **Segmentation and Classification**: In this phase, the final image undergoes segmentation using the K-means algorithm. Subsequently, a very basic classification approach is employed: each segment's area (colored pixels over black pixels) is utilized to differentiate regions. Typically, the analyzed segments exhibit a relatively consistent number of colored pixels, facilitating classification. This classification process organizes segments based on size, beginning with the smallest:

   1. Cerebrospinal fluid
   2. Grey matter
   3. White matter

   ```python
   # Perform segmentation using K-means
   images = segmentation.kmeans_segmentation(src1=img, src2=brain)
   ```

5. **3D plot**: 

The proposed functions are all publicly accessible. The other functions should be considered private, as they are only support functions used internally. 

> [!NOTE]
> The full documentation can be found [here](https://christiansassi.github.io/signal-image-and-video-project/).

# Getting started

1. Initialize the workspace:

   ```bash
   git clone https://github.com/christiansassi/signal-image-and-video-project.git
   cd signal-image-and-video-project
   ```

2. Install all the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run [main.py](main.py). This example will segment some of the images within the dataset.

   ```
   python3 main.py
   ```

> [!WARNING]
> The current implementation in `main.py` focuses on a specific range of axial brain images. This excludes the uppermost and bottommost slices due to two challenges. Firstly, these end sections often contain very small portions of the brain, which are less informative for segmentation purposes. Secondly, these areas can be more complex and may require advanced image processing techniques beyond the scope of this project.

# Contacts

Pietro Bologna - [pietro.bologna@studenti.unitn.it](mailto:pietro.bologna@studenti.unitn.it)

Christian Sassi - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<a href="https://www.unitn.it/"><img src="assets/extras/unitn-logo.png" width="300px"></a>
