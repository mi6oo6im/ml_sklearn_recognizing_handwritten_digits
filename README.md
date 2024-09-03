## Handwritten Digit Recognition

This project demonstrates how to recognize handwritten digits using machine learning techniques. The project is based on a tutorial from Scikit-learn and utilizes the digits dataset for training and testing a classifier. The objective of this project is to serve as a steb-by-step tutorial and reference manual.

Project Overview
The goal of this project is to classify images of handwritten digits (0-9) using a machine learning model. We will use the Scikit-learn library in Python to:

* Load the digits dataset
* Preprocess the data
* Train a classifier
* Evaluate the model's performance
* Visualize some predictions
* Test with manually created example using MS Paint
Dataset
The dataset used is the digits dataset provided by Scikit-learn. It contains 1,797 samples of 8x8 pixel grayscale images of handwritten digits. Each image is represented as a flattened array of 64 features.

Technologies Used:
* Python 3.11
* MS Windows 10
* MS Paint

Python Libraries:
* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* pickle
* skimage
* PIL

To run the project, you'll need to have Python and the required libraries installed. You can install the necessary libraries using pip or Conda.

Run the Python script:
Execute the script to train the model and visualize the results.

Load the Dataset:
The digits dataset is loaded from Scikit-learn's datasets module.

Preprocess the Data:
The data is split into training and testing sets to evaluate the model's performance.

Train the Classifier:
A simple classifier, such as a Support Vector Machine (SVM), is trained on the training data.

Evaluate the Model:
The model is evaluated on the test data, and accuracy metrics are calculated.

Visualize Predictions:
The project visualizes some of the test images along with their predicted labels to show the model's performance.

Results:
The classifier is expected to achieve a high accuracy rate, typically above 90%. Example predictions are displayed to showcase how well the model can recognize handwritten digits.

Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, feel free to create a pull request or open an issue.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments:<br>
Scikit-learn Documentation: <br>
* [Handwritten Digits Recognition Tutorial](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)<br>
* [Metrics and scoring: classification-report](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)<br>
* [Metrics and scoring: Confusion matrix](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)<br>
* [The Digit Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#sphx-glr-auto-examples-datasets-plot-digits-last-image-py)<br>
* [Scikit-learn datasets](https://scikit-learn.org/stable/api/sklearn.datasets.html)<br>
* [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)<br>
* [Support Vector Machines: SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)<br>
* [Train-Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)<br>
* [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)<br>

Pithon Image Library Documentation: <br>
* [Pillow Image](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)<br>

Python Documentation: <br>
* [Zip](https://docs.python.org/3.3/library/functions.html#zip)<br>
