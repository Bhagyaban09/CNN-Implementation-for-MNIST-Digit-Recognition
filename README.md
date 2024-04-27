CNN Handwritten Digit Recognition
This repository contains a TensorFlow implementation of a Convolutional Neural Network (CNN) designed to recognize and classify handwritten digits from the UCI Optical Recognition of Handwritten Digits dataset. The project demonstrates the application of CNNs in image recognition, emphasizing their effectiveness in classifying images into distinct categories based on visual content.

Project Structure
model.py: Contains the CNN model architecture, training, and evaluation code.
data_preparation.py: Script for loading and preprocessing the dataset.
utils.py: Helper functions for model evaluation, such as generating a confusion matrix and classification report.
requirements.txt: Lists all the necessary Python libraries for the project.
notebooks/: Jupyter notebooks for exploratory data analysis and incremental testing of the model.
images/: Directory containing visual outputs like plots and confusion matrices for reference.
Dataset
The project uses the Optical Recognition of Handwritten Digits dataset from the UCI Machine Learning Repository. This dataset includes pre-processed images of handwritten digits, each of which is an 8x8 pixel image represented as an array of grayscale values.

Installation
Clone this repository to your local machine using:

bash
Copy code
git clone https://github.com/your-username/cnn-digit-recognition.git
Navigate into the project directory:

bash
Copy code
cd cnn-digit-recognition
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
To run the training script, execute:

bash
Copy code
python model.py
For exploratory data analysis and step-by-step execution, open the Jupyter notebooks located in the notebooks/ directory:

bash
Copy code
jupyter notebook notebooks/exploratory_analysis.ipynb
Model Architecture
The CNN model consists of the following layers:

Conv2D Layer: Filters the 8x8 input image with 32 kernels of size 3x3.
MaxPooling2D Layer: Applies max pooling with a pool size of 2x2.
Dropout Layer: Applied after max pooling with a dropout rate of 0.25.
Flatten Layer: Flattens the output from the convolutional layers.
Dense Layer: A fully connected layer with 128 units and ReLU activation.
Output Layer: A softmax layer with 10 units corresponding to the 10 digit classes.
Results
The model achieves an average accuracy of approximately 98.7% across 5-fold cross-validation. Detailed results including precision, recall, and F1-scores for each class are provided in the images/ directory.

Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests to the main branch. For major changes, please open an issue first to discuss what you would like to change.

Ensure to update tests as appropriate.


Contact
Your Name - bhagyaban30999@gmail.com
