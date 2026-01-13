"""
Main Script - Complete ML Pipeline
Trains models, evaluates them, and generates visualizations
"""

import os
from train_models import ModelTrainer
from preprocessing import TextPreprocessor, FeatureExtractor
from evaluate_visualize import generate_all_visualizations, print_detailed_report

def main():
    """Main function to run the complete pipeline"""
    
    print("="*60)
    print("Text Classification System - ML Pipeline")
    print("="*60)
    
    # Configuration
    data_file = 'data/sentiment_data.csv'
    text_column = 'text'
    label_column = 'label'
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"\nError: Data file '{data_file}' not found!")
        print("Please run 'python generate_sample_data.py' first to create sample data.")
        print("Or provide your own dataset in the same format.")
        return
    
    # Initialize components
    print("\n1. Initializing components...")
    preprocessor = TextPreprocessor(remove_stopwords=True, stem_words=True)
    feature_extractor = FeatureExtractor(method='tfidf', max_features=5000, ngram_range=(1, 2))
    trainer = ModelTrainer(preprocessor, feature_extractor)
    
    # Load data
    print("\n2. Loading data...")
    texts, labels = trainer.load_data(data_file, text_column, label_column)
    print(f"   Loaded {len(texts)} samples")
    print(f"   Classes: {set(labels)}")
    
    # Prepare data
    print("\n3. Preprocessing and feature extraction...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(texts, labels)
    
    # Train all models
    print("\n4. Training models...")
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Save models
    print("\n5. Saving models...")
    trainer.save_models()
    
    # Generate visualizations
    print("\n6. Generating visualizations...")
    generate_all_visualizations(trainer.results)
    
    # Print detailed reports
    print("\n7. Detailed classification reports:")
    print_detailed_report(trainer.results)
    
    # Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    best_name, best_model, best_metrics = trainer.get_best_model()
    print(f"\nBest Model: {best_name}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
    
    print("\nNext steps:")
    print("1. View visualizations in the 'results' directory")
    print("2. Run 'streamlit run app.py' to start the web interface")
    print("3. Use the trained models for predictions")

if __name__ == '__main__':
    main()
