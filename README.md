# Text Classification System

A comprehensive machine learning system for text classification that demonstrates the complete ML pipeline from data preprocessing to model deployment. This project implements multiple classification models (Naive Bayes, Logistic Regression, and SVM) and provides a user-friendly web interface for predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Choices](#model-choices)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

## Overview

This project implements a text classification system that can categorize text data into predefined categories. The system includes:

- **Text Preprocessing**: Cleaning, tokenization, stopwords removal, and stemming
- **Feature Extraction**: TF-IDF vectorization for converting text to numerical features
- **Multiple Models**: Naive Bayes, Logistic Regression, and SVM
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- **Visualizations**: Performance comparison charts and confusion matrices
- **Web Interface**: Streamlit-based interactive interface for predictions

## Features

- **Complete ML Pipeline**: From raw text to predictions
- **Multiple Models**: Compare Naive Bayes, Logistic Regression, and SVM
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score metrics
- **Visualizations**: Model comparison charts and confusion matrices
- **Interactive Web Interface**: Easy-to-use Streamlit application
- **Extensible Design**: Easy to add new models or datasets

## Dataset

The project includes a sample sentiment analysis dataset with three categories:
- **Positive**: Positive sentiment texts
- **Negative**: Negative sentiment texts
- **Neutral**: Neutral sentiment texts

The dataset contains 60 samples (20 per class) and can be easily replaced with your own dataset.

### Dataset Format

Your dataset should be a CSV file with the following columns:
- `text`: The text content to classify
- `label`: The category/class label

Example:
```csv
text,label
"I love this product!",positive
"This is terrible.",negative
"It's okay.",neutral
```

## Approach

### 1. Text Preprocessing

The preprocessing pipeline includes:
- **Text Cleaning**: Remove URLs, emails, special characters, convert to lowercase
- **Tokenization**: Split text into individual words
- **Stopwords Removal**: Remove common words (the, is, at, etc.)
- **Stemming**: Reduce words to their root form (running → run)

### 2. Feature Extraction

- **TF-IDF Vectorization**: Converts text to numerical features
  - Term Frequency-Inverse Document Frequency weighting
  - Captures importance of words in documents
  - Max features: 5000
  - N-gram range: (1, 2) - includes unigrams and bigrams

### 3. Model Training

Three models are trained and compared:
- **Naive Bayes**: Fast probabilistic classifier
- **Logistic Regression**: Linear classifier with probability estimates
- **SVM**: Support Vector Machine with linear kernel

### 4. Evaluation

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### 5. Visualization

- Model performance comparison bar charts
- Confusion matrices for each model
- Results summary table

## Model Choices

### Naive Bayes
**Why chosen:**
- Fast training and prediction
- Works well with high-dimensional text data
- Probabilistic approach provides confidence scores
- Good baseline for text classification
- Assumes feature independence (naive assumption), which often works well for text

**Use case**: Quick predictions, baseline model, when interpretability is needed

### Logistic Regression
**Why chosen:**
- Provides probability estimates
- Interpretable coefficients
- Good baseline for classification tasks
- Handles multi-class classification well
- Regularization helps prevent overfitting

**Use case**: When you need interpretability, probability estimates, or a strong baseline

### SVM (Support Vector Machine)
**Why chosen:**
- Effective with high-dimensional data (like text)
- Finds optimal decision boundary
- Good generalization performance
- Linear kernel works well for text classification
- Can handle non-linear relationships with different kernels

**Use case**: When you need strong performance, especially with high-dimensional sparse data

## Project Structure

```
Text Classification System/
│
├── data/
│   └── sentiment_data.csv          # Dataset (generated or provided)
│
├── models/                          # Trained models (generated)
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── preprocessor.joblib
│   └── feature_extractor.joblib
│
├── results/                         # Evaluation results (generated)
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── results_table.csv
│
├── preprocessing.py                 # Text preprocessing module
├── train_models.py                  # Model training module
├── evaluate_visualize.py            # Evaluation and visualization
├── app.py                           # Streamlit web interface
├── main.py                          # Main pipeline script
├── generate_sample_data.py          # Sample data generator
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project

```bash
cd "Text Classification System"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- pandas
- scikit-learn
- nltk
- streamlit
- matplotlib
- seaborn
- joblib

### Step 4: Generate Sample Data (Optional)

If you want to use the sample dataset:

```bash
python generate_sample_data.py
```

This creates `data/sentiment_data.csv` with sample sentiment analysis data.

### Step 5: Prepare Your Dataset (Alternative)

If you have your own dataset:
1. Create a `data` directory
2. Place your CSV file as `data/sentiment_data.csv`
3. Ensure it has `text` and `label` columns

## Usage

### Training Models

Run the complete ML pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess the data
2. Extract features using TF-IDF
3. Train all three models
4. Evaluate and compare models
5. Generate visualizations
6. Save trained models

### Using the Web Interface

Start the Streamlit application:

```bash
streamlit run app.py
```

The web interface will open in your browser. You can:
- Enter text to classify
- Select different models
- View predictions and confidence scores
- Compare predictions across models

### Individual Components

You can also use individual modules:

```python
# Train models only
python train_models.py

# Generate sample data
python generate_sample_data.py
```

## Results

After training, you'll find:

1. **Model Comparison Chart** (`results/model_comparison.png`)
   - Bar chart comparing all metrics across models

2. **Confusion Matrices** (`results/confusion_matrices.png`)
   - Normalized confusion matrices for each model

3. **Results Table** (`results/results_table.csv`)
   - CSV file with all metrics for easy comparison

4. **Console Output**
   - Detailed metrics for each model
   - Classification reports
   - Best model identification

### Expected Performance

With the sample dataset:
- **Naive Bayes**: Typically achieves 85-95% accuracy
- **Logistic Regression**: Typically achieves 90-95% accuracy
- **SVM**: Typically achieves 90-95% accuracy

*Note: Actual performance depends on your dataset size and quality.*

## Observations

Based on typical results:

1. **All models perform well** on text classification tasks with proper preprocessing
2. **Logistic Regression and SVM** often achieve similar performance
3. **Naive Bayes** is faster but may have slightly lower accuracy
4. **TF-IDF features** effectively capture important words and phrases
5. **N-grams (bigrams)** help capture context and improve performance
6. **Stopwords removal** reduces noise and improves model focus
7. **Stemming** helps handle word variations (running, runs → run)

## Technologies Used

- **Python 3.8+**: Programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
  - MultinomialNB (Naive Bayes)
  - LogisticRegression
  - SVC (Support Vector Machine)
  - TfidfVectorizer
  - train_test_split
- **NLTK**: Natural language processing
  - Tokenization
  - Stopwords
  - Stemming
- **Streamlit**: Web interface framework
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization

## Notes

- The system is designed to be extensible - you can easily add new models or preprocessing steps
- For production use, consider:
  - Larger, more diverse datasets
  - Hyperparameter tuning
  - Cross-validation
  - Model versioning
  - API deployment instead of Streamlit

## Contributing

Feel free to extend this project:
- Add more models (Random Forest, Neural Networks, etc.)
- Implement different feature extraction methods (Word2Vec, BERT, etc.)
- Add hyperparameter tuning
- Improve the web interface
- Add more visualizations

## License

This project is for educational purposes.

---

**Happy Classifying! 🎉**
