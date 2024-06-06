# Representation Learning of Histopathology Images using Graph Neural Networks
### Paper Abstract
Representation learning for Whole Slide Images (WSIs) is pivotal in developing image-based systems to achieve higher precision in diagnostic pathology. We propose a two-stage framework for WSI representation learning. We sample relevant patches using a color-based method and use graph neural networks to learn relations among sampled patches to aggregate the image information into a single vector representation. We introduce attention via graph pooling to automatically infer patches with higher relevance. We demonstrate the performance of our approach for discriminating two sub-types of lung cancers, Lung Adenocarcinoma (LUAD) & Lung Squamous Cell Carcinoma (LUSC). We collected 1,026 lung cancer WSIs with the 40× magnification from The Cancer Genome Atlas (TCGA) dataset, the largest public repository of histopathology images and achieved state-of-the-art accuracy of 88.8% and AUC of 0.89 on lung cancer sub-type classification by extracting features from a pre-trained DenseNet model.
### Useful Links
- [Read the paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w57/Adnan_Representation_Learning_of_Histopathology_Images_Using_Graph_Neural_Networks_CVPRW_2020_paper.pdf)
- [Learn more on Kimia Lab](https://kimialab.uwaterloo.ca/kimia/index.php/representation-learning-of-histopathology-images-using-graph-neural-networks/)
### Disclaimer
Kimia Lab at Mayo Clinic does not own the code in this repository. The code and data were produced in Kimia Lab at the University of Waterloo. The code is provided as-is without any guarantees, and is stored here as part of Kimia Lab's history. We welcome questions and comments.

This code is intended for research purposes only. Before using or cloning this repository, please read the [End User Agreement](agreement.pdf).
