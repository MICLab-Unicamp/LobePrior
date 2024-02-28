# LobePrior: Segmenting lung lobes in CT images in the presence of severe abnormalities

Project developed for XXIX Congresso Brasileiro de Engenharia Biomédica (CBEB) 2024 (https://sbeb.org.br/cbeb2024)

## Abstract

> The development of efficient and robust algorithms in automated tools for segmenting the lung and its lobes is useful for the diagnosis and monitoring of lung diseases that cause lung abnormalities, such as pneumonia caused by COVID-19 and cancerous nodules. The amount of available data containing manual annotations of the lobes in patients with severe lung abnormalities such as consolidations and ground glass opacities is scarce, due to the difficulty of visualizing the lobar fissures. This work aims to develop a method for automated segmentation of lung lobes using deep neural networks in computed tomography images of the lungs of patients with severe abnormalities. The method is based on probabilistic models built from labels fusion, used not only to guide the deep neural networks while learning to segment the lobes, but also for postprocessing the network prediction to obtain the final segmentation. Segmentation is performed in two stages: a coarse stage working on downsampled images and a second high-resolution stage, where specialized AttUNets compete for each lobe's segmentation. The performance of the proposed approach was assessed using two public datasets with lobe annotations, in the presence of cancer nodules and COVID-19 consolidations. Open source implementation is available at

# To install dependencies
> sh requirements.txt

# To run project
> python main.py

<!--
We present an approach using probabilistic models, constructed from lung CT images. The images were recorded and separated into groups, according to shape and appearance. The images were separated into groups because of the great difference between the shapes that the lung has between patients. Added to post-processing and templates, a model capable of segmenting CT images of lungs affected by severe diseases was developed. The main contribution of this work was to improve the quality of these segmentations and present a model capable of identifying lobar fissures more efficiently, as this is a task considered very difficult, overcoming the difficulty of the methods in finding the fissures correctly, as they are healthy. deformed by lung diseases such as cancer and COVID-19.
-->

# Project

   * This work was submitted to ISBI 2024 (https://biomedicalimaging.org/2024).

   <center>
	<figure>
	    <img src="https://github.com/MICLab-Unicamp/LobePrior/blob/main/images/Lobes_coronacases_001.png", alt="Unet", height="300" width="300">
	</figure>
   </center>

<!-- * Implemented network diagram

<center>
	<figure>
	    <img src="https://github.com/MICLab-Unicamp/LobePrior/blob/main/images/Model_fusion_vertical.png" alt="Unet", height="300" width="300">
	</figure>
</center>
-->

* Example of segmentation of the LobePrior network in the CT image coronacases_007

   <center>
	<figure>
	    <img src="https://github.com/MICLab-Unicamp/LobePrior/blob/main/images/lobeprior.png" alt="Unet", height="300" width="300">
	</figure>
   </center>
