	COVID-19 Detection Using Deep Learning on
Chest X-Ray Images: An Applied Machine
Learning Approach
Applied Machine Learning
Table of Contents
Introduction ...........................................................................................2
Problem Statement ..................................................................................2
Dataset ..................................................................................................3
Dataset description.................................................................................................3
File Description:.........................................................................................................................3
Context......................................................................................................................................3
Content .....................................................................................................................................3
Acknowledgements....................................................................................................................3
Inspiration .................................................................................................................................4
Pre-processing steps.............................................................................4
1- Data Augmentation: ...........................................................................................................4
2- Normalization: ...................................................................................................................4
3- Resizing:............................................................................................................................4
4- Splitting the Dataset:..........................................................................................................5
5- Label Encoding: .................................................................................................................6
Methodology .........................................................................................6
The training process .............................................................................7
Optimizer:..................................................................................................................................7
Loss function: ............................................................................................................................7
Epochs and Batch Size: ..............................................................................................................7
Results and performance evaluation......................................................8
Comparison Between Models................................................................................8
Discussion and Interpretation of Results...................................................................................10
Conclusion ..............................................................................................................................10
Difficulties we faced:........................................................................... 11
References Used in the Project............................................................ 11
Introduction
The COVID-19 pandemic, caused by the SARS-CoV-2 virus, has posed unprecedented
challenges to global health systems. Rapid and accurate diagnosis is crucial for controlling the
spread of the virus and for timely treatment of affected individuals. While RT-PCR tests are the
gold standard for COVID-19 diagnosis, they come with limitations such as longer processing
times and limited availability in certain regions. Consequently, there is a need for supplementary
diagnostic tools that can provide quick and accurate results.
Medical imaging, particularly chest X-rays (CXR), has emerged as a valuable diagnostic tool in
the context of COVID-19. Chest X-rays are widely available, cost-effective, and can be rapidly
performed, making them an excellent candidate for initial screening and diagnosis. However,
interpreting chest X-ray images requires significant expertise, and manual evaluation can be
time-consuming and subject to human error.
This project aims to leverage the power of deep learning to develop automated methods for
classifying chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal.
By employing advanced machine learning models, we seek to create a tool that can assist
radiologists and healthcare professionals in diagnosing COVID-19 more efficiently and
accurately.
Problem Statement
The main objective of this project is to develop a machine learning-based classification model
that can automatically analyze chest X-ray images and accurately categorize them into one of the
following classes:
1. COVID-19: X-ray images showing lung abnormalities specific to COVID-19 infection.
2. Viral Pneumonia: X-ray images showing lung abnormalities caused by other viral
infections.
3. Normal: X-ray images of healthy lungs without any signs of infection.
The key challenges addressed in this project include:
• Data Quality and Quantity: Ensuring that the dataset used is of high quality and has
sufficient examples for each class to train robust models.
• Feature Extraction: Developing models that can effectively extract and utilize features
from chest X-ray images to distinguish between the different categories.
• Model Performance: Achieving high accuracy, precision, recall, and F1-score in
classification to ensure the model's reliability in clinical settings.
• Generalization: Ensuring that the model generalizes well to new, unseen data, and is not
overfitted to the training dataset.
By addressing these challenges, the project aims to create a reliable, automated tool that can aid
in the early detection and diagnosis of COVID-19, thereby contributing to better patient
outcomes and more efficient healthcare delivery.
Dataset
The data set we used was from kaggle,
https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset/data
Dataset description
It contains around 137 cleaned images of COVID-19 and 317 in total containing Viral
Pneumonia and Normal Chest X-Rays structured into the test and train directories.
The train directory has 3 files: covid, normal and viral pneumonia.
Covid: 111 images
Normal: 70 images
Viral pneumonia: 70 images
The test directory contains 3 files as well covid, normal and viral pneumonia.
Covid: 26 images
Normal: 20images
Viral pneumonia: 20 images
File Description:
- The covid files contain images that show the characteristic signs of COVID-19
pneumonia including bilateral multilobar ground glass opacities and consolidation
in the lungs.
- The normal cases have images that show clear lungs without any signs of
infection or inflammation. They are used as the control group to distinguish
COVID-19 from normal radiographs.
- The viral pneumonia file has images showing signs of pneumonia caused by other
viruses like influenza. They help differentiate COVID-19 from pneumonia of
non-COVID viral origin.
Context
Helping Deep Learning and AI Enthusiasts to contribute to improving COVID-19 detection
using just Chest X-rays.
Content
It is a simple directory structure branched into test and train and further branched into the
respective 3 classes which contains the images.
Acknowledgements
The University of Montreal for releasing the images.
Inspiration
Help the medical and researcher community by sharing my work and encourage them to
contribute extensively.
Pre-processing steps
Pre-processing is an essential step in preparing the dataset for training machine learning models.
Below are the detailed pre-processing steps and their implementation in our project:
1- Data Augmentation:
Purpose: To artificially increase the size of the training dataset and improve the model's ability
to generalize.
Techniques Used:
Rotation: Randomly rotating the images by a certain angle (e.g., -30 to 30 degrees).
Zooming: Randomly zooming in on the images.
Horizontal Flipping: Flipping the images horizontally.
Shearing: Applying random shearing transformations.
Shifting: Randomly shifting the images along the width and height.
Implementation: Data augmentation was implemented using Keras' ImageDataGenerator. This
step was applied to the training dataset to increase the diversity of the input images.
2- Normalization:
Purpose: To scale the pixel values of the images to a standard range, typically [0, 1], which
helps in faster convergence during training.
Technique: Pixel values were divided by 255 to normalize them from a range of [0, 255] to
[0, 1].
Implementation: Normalization was applied to all datasets (training, validation, and test).
3- Resizing:
Purpose: To ensure that all images have the same dimensions, which is required for batch
processing in neural networks.
Dimensions: Images were resized to 224x224 pixels, matching the input size expected by
models like VGG.
Implementation: Resizing was performed using OpenCV or PIL. This step was applied to all
images in the dataset.
4- Splitting the Dataset:
Purpose: To create separate sets for training, validation, and testing, which is essential for
evaluating model performance.
Technique:
Training Set: Used for training the model.
Validation Set: Used for tuning model parameters and avoiding overfitting.
Test Set: Used for the final evaluation of model performance.
Implementation: The dataset was split using a stratified sampling approach to ensure each set
had a representative distribution of the classes.
5- Label Encoding:
Purpose: To convert categorical labels (e.g., COVID-19, Viral Pneumonia, Normal) into
numerical values that can be used by the machine learning models.
Technique: Labels were encoded using one-hot encoding, resulting in a binary matrix
representation where each class is represented by a unique binary vector.
Implementation: One-hot encoding was implemented using TensorFlow or Scikit-learn.
Methodology
In this project, we employed several machine learning models to classify chest X-ray images into
three categories: COVID-19, Viral Pneumonia, and Normal.
We implemented 5 models:
1- VGG16
2- VGG19
3- Customized Convolutional Neural Network (CNN)
4- Deep Neural Network (DNN)
VGG16 and VGG19
• Architecture: VGG16 and VGG19 are deep convolutional neural networks with 16 and
19 layers, respectively. They use small 3x3 filters throughout the entire network and
employ max-pooling layers.
• Advantages: These models are known for their simplicity and effectiveness in image
classification tasks.
• Implementation: We used the pre-trained VGG16 and VGG19 models from Keras'
applications module, and fine-tuned them for our specific classification task.
Customized CNN
• Architecture: A custom-designed convolutional neural network tailored specifically for
this project. The architecture consists of several convolutional layers followed by maxpooling layers, fully connected layers, and dropout for regularization.
• Advantages: A customized architecture can be optimized for the specific characteristics
of the dataset.
• Implementation: We built the custom CNN from scratch using Keras.
Deep Neural Network (DNN)
• Architecture: A fully connected deep neural network with multiple hidden layers.
• Advantages: Serves as a baseline model to compare with convolutional architectures.
• Implementation: We implemented the DNN using Keras' Sequential API.
The training process
Optimizer:
▪ The Adam optimizer is used to train the models. Adam is a commonly
used optimization algorithm for training neural networks.
Loss function:
▪ Categorical crossentropy loss is used for models with a single softmax
output layer, like VGG16, VGG19 and AlexNet.
Epochs and Batch Size:
▪ A batch size of 32 was likely used for all models since it is the Keras
default .
▪ For every model the epochs differed to get the best accuracy, the epochs ranged from 10-50
epochs in our models.
Results and performance evaluation
Model Test
Loss
Test
Accuracy
F-1
Score Percision Recall
VGG 16 0.06 0.97 0.97 0.97 0.97
VGG 19 0.19 0.94 0.94 0.94 0.94
Customized
CNN 0.70 0.88 0.41 0.41 0.42
DNN 0.37 0.83 0.83 0.85 0.83
Comparison Between Models
1. VGG16
• Test Loss: 0.06
• Test Accuracy: 0.97
• F1-Score: 0.97
• Precision: 0.97
• Recall: 0.97
Analysis:
• VGG16 achieved the best performance among all models with the highest test accuracy
(0.97) and F1-Score (0.97).
• The low test loss (0.06) indicates that the model generalizes well to the unseen test data.
• High precision and recall values (0.97) indicate that the model has a balanced
performance in correctly identifying all classes.
2. VGG19
• Test Loss: 0.19
• Test Accuracy: 0.94
• F1-Score: 0.94
• Precision: 0.94
• Recall: 0.94
Analysis:
• VGG19 also performed well but slightly less than VGG16.
• The test accuracy (0.94) and F1-Score (0.94) are still high, indicating a strong
performance.
• The higher test loss (0.19) compared to VGG16 suggests that it may not generalize as
well as VGG16.
3. Customized CNN
• Test Loss: 0.70
• Test Accuracy: 0.88
• F1-Score: 0.41
• Precision: 0.41
• Recall: 0.42
Analysis:
• The Customized CNN showed a significantly lower performance compared to VGG16
and VGG19.
• The test accuracy (0.88) is lower, and the F1-Score (0.41) indicates poor precision and
recall.
• The high test loss (0.70) suggests that the model struggles with the classification task and
does not generalize well.
4. DNN
• Test Loss: 0.37
• Test Accuracy: 0.83
• F1-Score: 0.83
• Precision: 0.85
• Recall: 0.83
Analysis:
• The DNN performed better than the Customized CNN but worse than VGG16 and
VGG19.
• The test accuracy (0.83) and F1-Score (0.83) indicate moderate performance.
• Precision (0.85) is slightly higher than recall (0.83), suggesting that the model is better at
identifying true positives than minimizing false negatives.
• The test loss (0.37) is relatively high, indicating that there is room for improvement in the
model's generalization capability.
Discussion and Interpretation of Results
• VGG16 and VGG19: Both pre-trained models demonstrated superior performance due
to their deep architectures and extensive pre-training on large datasets like ImageNet.
Fine-tuning these models for the specific task of chest X-ray classification allowed them
to achieve high accuracy, precision, recall, and F1-Score. VGG16 outperformed VGG19,
which might be due to its slightly simpler architecture leading to better generalization on
this dataset.
• Customized CNN: The custom CNN model underperformed compared to the pre-trained
models. This suggests that the architecture was not as effective for this specific
classification task, potentially due to insufficient complexity or suboptimal
hyperparameters.
• DNN: The deep neural network, lacking convolutional layers, performed worse than the
convolutional models. This result underscores the importance of convolutional layers in
capturing spatial hierarchies in image data.
Conclusion
• Best Model: VGG16 emerged as the best-performing model, achieving the highest
accuracy and F1-Score, along with excellent precision and recall. This model is
recommended for deployment in applications requiring chest X-ray image classification.
• Future Improvements: Further improvements could be made by experimenting with
more advanced architectures such as EfficientNet or DenseNet, implementing additional
data augmentation techniques, and optimizing hyperparameters through more extensive
grid searches.
Difficulties we faced:
The main difficulty we faced was achieving a high accuracy in the models, how we overcame
that was different between each model. Some models we had to increase epochs, others we
needed to alter in the batch size. We also did data augmentation, hyperparameter tuning, early
stopping, learning rate scheduling, regularization techniques and fine tuning.
References Used in the Project
1. Keras Documentation: https://keras.io/api/applications/
2. ImageDataGenerator Documentation: https://keras.io/api/preprocessing/image/
3. VGG Paper: Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional
Networks for Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
4. TensorFlow Documentation: https://www.tensorflow.org/api_docs
5. Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html