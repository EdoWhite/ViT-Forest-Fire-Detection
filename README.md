# Transformers-Based Smoke and Fire Detection in Forest Environments

## Overview

The proposed system consists of **two main solutions**. In the first solution we design, train, and evaluate a **custom CNN**. Regarding the second solution, we will fine-tune a pretrained **vision transformer**, the Google ViT [9], hosted on the Hugging Face Hub [8]. Results are logged and proposed through WanDB [7]. The performance of the two solutions will be compared and different aspects will be discussed.

The framework used for this experiment is PyTorch with the PyTorch Lightning Module. The final web application to showcase the model is written in Python using the Streamlit library and hosted on the Hugging Face Hub.

## Data Sources

The dataset used in this project is **self-built** by merging two **datasets from Kaggle**. In particular, we used samples from ”train_fire” and ”train_smoke” from [5], and all the samples (mixed together from further splitting) from [6].

The final dataset consists of **2525 samples for each of the three classification classes** (fire, smoke, normal). This dataset will be then split into train, validation, and test sets. The images in the dataset depict **fires of different magnitude**, **from different perspectives** (aerial, far, near) thus **simulating images from different devices**, such as webcams and drones.

## Demo Web Application
<img src="./IMG/WebApp.png">
To showcase the final model, we developed a simple web application, available at https://huggingface.co/spaces/EdBianchi/Forest-Fire-Detection The demo allows the user to upload an image and predict if there is a smoke, fire or normal situation. We report here a screenshot of the GUI.

## References
[1] European forest fire report: Three of the worst fire seasons on record took place in the last six years[J]. 2022.

[2] Pennazza G, Seydi S T, Saeidi V, et al. Fire-net: A deep learning framework for active forest fire detection[J]. Journal of Sensors, 2022, 2022: 8044390. http://dx.doi.org/10.1155/2022/ 8044390.

[3] Jiao Z, Zhang Y, Xin J, et al. A deep learning based forest fire detection approach using uav and yolov3.

[4] Vadakkadathu Rajan G, Paul S. Forest fire detection using machine learning[J]. 2022.

[5] KUTLU K. Forest fire dataset on kaggle[EB/OL]. https://www.kaggle.com/datasets/kutaykutlu/forest-fire?select=train_fire.

[6] PRASAD M S. Forest fire images dataset on kaggle[EB/OL]. https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images. 

[7] Biewald L. Experiment tracking with weights and biases. Software available from wandb.com.

[8] Hugging face[EB/OL]. https://huggingface.co.

[9] B. Wu et al., Visual Transformers: Token-based Image Representation and Processing for Computer Vision. arXiv, 2020. doi: 10.48550/ARXIV.2006.03677.

[10] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, RandAugment: Practical automated data augmentation with a reduced search space. arXiv, 2019. doi: 10.48550/ARXIV.1909.13719.

[11] A. Steiner, A. Kolesnikov, X. Zhai, R. Wightman, J. Uszkoreit, and L. Beyer, “How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers,” 2021, doi: 10.48550/ARXIV.2106.10270.

[12] A. Nguyen, K. Pham, D. Ngo, T. Ngo, and L. Pham, An Analysis of State-of-the-art Activation Functions For Supervised Deep Neural Network. arXiv, 2021. doi: 10.48550/ARXIV.2104.02523.
