# IMAGE-CLASSIFICATION-MODEL

COMPANY : CODETECH IT SOLUTIONS

NAME : ANAS AHMAD

INTERN ID : CT06DG87

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

The task involves developing an image classification model using a Convolutional Neural Network (CNN), leveraging either TensorFlow or PyTorch, to accurately classify images from a dataset. The goal is to build a deep learning-based image classifier that can learn from a training dataset, evaluate its performance on a test dataset, and produce performance metrics such as accuracy and loss. In this project, the TensorFlow library is used due to its simplicity and extensive community support. The model is built and executed using the PyCharm IDE, which provides robust Python development features including virtual environment support and built-in terminal. The image dataset used is CIFAR-10, a widely used benchmark dataset consisting of 60,000 32x32 color images in 10 different classes, such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is preprocessed by normalizing the pixel values to range between 0 and 1. The CNN model is structured with three convolutional layers: the first layer uses 32 filters, the second and third use 64 filters each, all with a kernel size of 3x3 and ReLU activation. Each convolutional block is followed by a max-pooling layer to reduce spatial dimensions. After the convolutional layers, the model is flattened and passed through a fully connected dense layer with 64 units and ReLU activation, followed by an output layer with 10 units (corresponding to the 10 classes) using a softmax function to produce class probabilities. The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss (suitable for integer labels), and accuracy as the evaluation metric. The model is then trained for 10 epochs with validation performed on the test dataset to monitor performance. After training, the model is evaluated on the test dataset to calculate final test accuracy and loss. A line plot is generated using matplotlib to visualize training and validation accuracy across epochs, helping to assess model learning and overfitting trends. Additionally, the trained model is used to make predictions on new images, and sample predictions are printed for verification. All required Python libraries such as TensorFlow, matplotlib, and numpy are installed via the terminal using pip commands. The entire project is implemented in a single Python file (`main.py`) within PyCharm, ensuring modularity and ease of execution. The deliverables for this task include the Python source code file, a trained CNN model (which can optionally be saved using `model.save()`), and the performance evaluation outputs such as accuracy metrics and accuracy plots. This project demonstrates practical skills in building and evaluating deep learning models, understanding of CNN architecture, and the ability to work with machine learning frameworks like TensorFlow in a development environment like PyCharm. Overall, it fulfills the objective of creating a functional, efficient, and well-evaluated image classification system.

OUTPUT : 
![image](https://github.com/ANASAHMAD-CLOUD/IMAGE-CLASSIFICATION-MODEL/blob/main/Figure_1.png)
