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
