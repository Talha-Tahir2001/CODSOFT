# Task 1

# Movie Genre Classification

## About

Movie Genre Classification is a machine learning project that predicts the genre of a movie based on its plot summary or other textual information. This project leverages natural language processing techniques and deep learning models to classify movies into various genres.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following dependencies installed:

- Python (>=3.6)
- TensorFlow
- scikit-learn
- Jupyter Notebook (for training and experimenting)

You can install the required Python libraries using the following command:

pip install tensorflow scikit-learn jupyter
### Installation

1.  Clone this repository to your local machine:

`git clone https://github.com/YourUsername/Movie-Genre-Classification.git
cd Movie-Genre-Classification` 

2.  Download the dataset and place it in the appropriate folder.
	The movie genre classification dataset used in this project is 	sourced from Kaggle. It includes a diverse collection of movie plot summaries and their corresponding genres.
-  **Dataset Source**: [Kaggle - Movie Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
-  **Description**: This dataset contains movie plot summaries from various genres, making it suitable for training and evaluating the genre classification model.
    
3.  Train the model using Jupyter Notebook.
    
## Usage

To use the Movie Genre Classification model, follow these steps:

1.  Ensure you have installed the necessary dependencies and have trained the model.

2.  After that save the model into a "H5" format.
-  `# Assuming 'model' is your trained model:`
-    model.save('your_model.h5')
    
4.  Load the model and integrate it into gradio or streamlit or whichever UI Library you prefer.
- `You can load the model using the following:`
- model = tf.keras.models.load_model('your_model.h5')

5.  The predicted genre(s) will be displayed on the web interface or UI.
    

## Features

-   Predicts multiple genres for a given movie plot summary.
-   Easily customizable to include additional features or improve model accuracy.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow our [contributing guidelines](https://chat.openai.com/c/CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE.md).

## Acknowledgments

-   Thanks to the creators of the dataset used in this project.
-   Grateful to the open-source community for providing useful libraries and tools.



# Task 2

# Spam SMS Detector

## Overview

The **Spam SMS Detector** is a machine learning project designed to automatically identify and filter out spam text messages. This Jupyter Notebook-based project leverages natural language processing (NLP) techniques and various classifiers to create an efficient and accurate spam detection system.

## Table of Contents

-   [Getting Started](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#getting-started)
    -   [Prerequisites](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#prerequisites)
    -   [Usage](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#usage)
-   [Model Training](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#model-training)
- [Files](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#files)
- [Gradio App](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#gradio-app)
-   [Contributing](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#contributing)
-   [License](https://chat.openai.com/c/274f2198-6926-43b4-bf68-d62325ee6e02#license)

## Getting Started

### Prerequisites

Before you begin, ensure you have:

-   [Jupyter Notebook](https://jupyter.org/install) installed.
-   Python 3.x and required libraries installed.
-  You can install the required libraries by using:
	`pip install pandas numpy matplotlib seaborn sklearn nltk pickle gradio`

### Usage

1.  Clone this repository:
    
    `git clone https://github.com/your-username/Spam-SMS-Detector.git` 
    
2.  Open the `Spam SMS Detector.ipynb` Jupyter Notebook file.
    
3.  Follow the instructions and run the code cells to explore the project, including data preprocessing, model training, and evaluation.
    
4.  Experiment with the classifiers, adjust parameters, and fine-tune the models as needed.
    
5.  Use the trained models to classify SMS messages as spam or not spam.
    

## Model Training

The Jupyter Notebook file (`Spam SMS Detection.ipynb`) contains detailed information on model training and evaluation. You can explore different classifiers, evaluate their performance, and make improvements as necessary.
## Files

-   `Spam SMS Detector.ipynb`: The Jupyter Notebook file containing the code and documentation for this project.
-   `vectorizer.pkl`: A serialized version of the text vectorizer used for feature extraction.
-   `Spam-SMS-Detector.pkl`: A serialized version of the trained machine learning model.
## Gradio App

The Gradio app provides an interactive and user-friendly way to utilize the Spam SMS Detector. You can easily input SMS texts and receive instant predictions on their authenticity.
You can check it out via:
https://huggingface.co/spaces/Talha-tahir666/Spam-SMS-Detection
## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bugfix.
3.  Commit your changes with descriptive commit messages.
4.  Push your changes to your fork.
5.  Create a pull request, detailing your changes and improvements.

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).
