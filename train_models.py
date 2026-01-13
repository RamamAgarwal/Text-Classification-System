"""
Model Training Module
Trains multiple classification models and saves them
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
from preprocessing import TextPreprocessor, FeatureExtractor

class ModelTrainer:
    """Class for training and comparing multiple models"""
    
    def __init__(self, preprocessor, feature_extractor):
        """
        Initialize model trainer
        
        Args:
            preprocessor: TextPreprocessor instance
            feature_extractor: FeatureExtractor instance
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.models = {}
        self.results = {}
        
    def load_data(self, filepath, text_column, label_column):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            text_column: Name of column containing text
            label_column: Name of column containing labels
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(filepath)
        
        # Remove rows with missing values
        df = df.dropna(subset=[text_column, label_column])
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return texts, labels
    
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Preprocess texts and split into train/test sets
        
        Args:
            texts: List of raw text strings
            labels: List of labels
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Preprocess texts
        print("Preprocessing texts...")
        preprocessed_texts = self.preprocessor.preprocess_corpus(texts)
        
        # Extract features
        print("Extracting features...")
        X = self.feature_extractor.fit_transform(preprocessed_texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_naive_bayes(self, X_train, y_train):
        """
        Train Naive Bayes model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\nTraining Naive Bayes model...")
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\nTraining Logistic Regression model...")
        model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        model.fit(X_train, y_train)
        return model
    
    def train_svm(self, X_train, y_train):
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\nTraining SVM model...")
        # Use linear kernel for text classification (faster and often effective)
        model = SVC(kernel='linear', random_state=42, probability=True)
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a model and return metrics
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate them
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        # Train Naive Bayes
        nb_model = self.train_naive_bayes(X_train, y_train)
        self.models['Naive Bayes'] = nb_model
        self.results['Naive Bayes'] = self.evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
        
        # Train Logistic Regression
        lr_model = self.train_logistic_regression(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
        
        # Train SVM
        svm_model = self.train_svm(X_train, y_train)
        self.models['SVM'] = svm_model
        self.results['SVM'] = self.evaluate_model(svm_model, X_test, y_test, 'SVM')
    
    def save_models(self, directory='models'):
        """
        Save all trained models and preprocessor
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filename = os.path.join(directory, f"{name.replace(' ', '_').lower()}.joblib")
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save preprocessor and feature extractor
        joblib.dump(self.preprocessor, os.path.join(directory, 'preprocessor.joblib'))
        joblib.dump(self.feature_extractor, os.path.join(directory, 'feature_extractor.joblib'))
        print("Saved preprocessor and feature extractor")
    
    def get_best_model(self):
        """
        Get the best model based on F1-score
        
        Returns:
            Tuple of (model_name, model, metrics)
        """
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        return best_model_name, self.models[best_model_name], self.results[best_model_name]


def main():
    """Main function to train models"""
    # Configuration
    data_file = 'data/sentiment_data.csv'
    text_column = 'text'
    label_column = 'label'
    
    # Initialize components
    preprocessor = TextPreprocessor(remove_stopwords=True, stem_words=True)
    feature_extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
    trainer = ModelTrainer(preprocessor, feature_extractor)
    
    # Load data
    print("Loading data...")
    texts, labels = trainer.load_data(data_file, text_column, label_column)
    print(f"Loaded {len(texts)} samples")
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(texts, labels)
    
    # Train all models
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    trainer.save_models()
    
    # Print best model
    best_name, best_model, best_metrics = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")


if __name__ == '__main__':
    main()
