# Amazon-ML-Challenge

## About Amazon ML Challenge
![](https://he-s3.s3.amazonaws.com/media/cache/0a/be/0abe8c0908dcb9e67941600739f6d651.png)</br>
Amazon ML Challenge is a two-stage competition where students from all engineering campuses across India will get a unique opportunity to work on Amazon’s dataset to bring in fresh ideas and build innovative solutions for a real-world problem statement. The top three winning teams will receive pre-placement interviews (PPIs) for ML roles at Amazon along with cash prizes and certificates.

## <b>Image Entity Extraction from Product Images</b></br>

## ML Challenge Stages
![amazon mll](https://github.com/user-attachments/assets/b77bb2ad-9316-419e-aa2c-36cbd046b544)

## Datset Link🔗
https://unstop.com/hackathons/amazon-ml-challenge-amazon-1100713

## Overview

This project aims to develop a machine-learning model for extracting entity values from product images. The goal is to automate the extraction of key product details such as weight, dimensions, and other attributes directly from images. This capability is crucial for digital marketplaces where product information is often incomplete or missing.

## Problem Statement

As digital marketplaces expand, many products lack detailed textual descriptions. This makes it essential to obtain key details directly from images. Our task is to build a model that can accurately identify and extract these details, providing valuable information for product listings.

## Full Train/Test dataset details:</br>

index: A unique identifier (ID) for the data sample.</br>
image_link: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg  To download images, use the download_images function from src/utils.py. See sample code in src/test.ipynb.</br>
group_id: Category code of the product.</br>
entity_name: Product entity name. For example, “item_weight”.</br>
entity_value: Product entity value. For example, “34 gram”.</br>
Note: For test.csv, you will not see the column entity_value as it is the target variable.</br>

## Team Members
1. [Roshni Kumari](https://github.com/RSN601KRI)
2. [Antima Mishra](https://github.com/antima-123bit)
3. [Raushan Kumar](https://github.com/raushan0422)
4. [Varshita R](https://www.linkedin.com/in/varshitha-r-616b15241/)

## Data Description

The dataset consists of the following files:

- `dataset/train.csv`: Training data with labels.
- `dataset/test.csv`: Test data without labels (for predictions).
- `dataset/sample_test.csv`: Sample test input file.
- `dataset/sample_test_out.csv`: Sample output file showing the correct format.

**Columns:**

- **index**: Unique identifier for each data sample.
- **image_link**: URL to download the product image.
- **group_id**: Category code of the product.
- **entity_name**: Name of the product entity (e.g., "item_weight").
- **entity_value**: Value of the product entity (e.g., "34 gram").

## Objective

Develop a machine learning model to extract entity values from product images and generate predictions in the format "x unit", where `x` is a float number and `unit` is one of the allowed units.

## Approach

1. **Data Preparation:**
   - **Image Downloading:** Use the `download_images` function from `src/utils.py` to fetch images.
   - **Preprocessing:** Normalize and resize images for model input.

2. **Feature Extraction:**
   - Use Convolutional Neural Networks (CNNs) for feature extraction. Consider architectures like ResNet, Inception, or EfficientNet.
   - Fine-tune pre-trained models if necessary.

3. **Entity Extraction:**
   - Implement a Multi-Label Classification model or object detection models (e.g., YOLO, Faster R-CNN) to identify and classify entities within images.

4. **Post-Processing:**
   - Format predictions to match the required output format and ensure predictions are invalid units as listed in `src/constants.py`.

## Evaluation

- **Metrics:** The performance will be evaluated based on the F1 score, which considers Precision and Recall.
- **Scoring Formula:**
  \[
  \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

![Amazon ML](https://github.com/user-attachments/assets/fdfacc36-6b3c-444a-b72a-09865cfc062a)

![output](https://github.com/user-attachments/assets/eea549b3-c5a1-46ce-9ee2-46f4a3452f71)

## Resources

- [TensorFlow Tutorial on CNNs](https://www.tensorflow.org/tutorials/images/cnn)
- [PyTorch Image Classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [YOLO (You Only Look Once)](https://pjreddie.com/darknet/yolo/)
- [Faster R-CNN Tutorial](https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py)
- [Precision, Recall, F1 Score Explained](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

## Files

- **`src/sanity.py`**: Ensures the final output file meets formatting requirements.
- **`src/utils.py`**: Contains functions for downloading images.
- **`src/constants.py`**: Lists allowed units for entity values.
- **`sample_code.py`**: Sample code for generating output files (optional).

## Submission

- Generate predictions for `dataset/test.csv` and format them according to `dataset/sample_test_out.csv`.
- Submit the `test_out.csv` file in the portal with the exact formatting.

## Conclusion

By automating the extraction of product details from images, this project aims to enhance data accuracy, improve efficiency, and provide a better user experience in digital marketplaces.

# Connect With Me
LinkedIn : https://www.linkedin.com/in/roshnikumari1/<br/>
Email : roshni06k2004@gmail.com<br/>
Twitter : www.twitter.com/RoshniK29147303</br>
Website : https://bento.me/roshnikri </br>
# Personal
Name: Roshni Kumari<br/>
University: Galgotias University, Noida(UP)

# Gratitude
Thank You, if you like it please leave a Star.⭐
