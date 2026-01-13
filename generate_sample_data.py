"""
Generate Sample Dataset for Text Classification
Creates a sentiment analysis dataset for demonstration
"""

import pandas as pd
import os

def generate_sample_data():
    """Generate a sample sentiment analysis dataset"""
    
    # Sample positive texts
    positive_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the best service I've ever used. Highly recommended!",
        "Excellent quality and fast delivery. Very satisfied with my purchase.",
        "Outstanding customer service. They went above and beyond!",
        "Great value for money. I'm very happy with this purchase.",
        "Perfect! Exactly what I was looking for. Thank you!",
        "Amazing experience! Will definitely buy again.",
        "Top-notch quality. Exceeded my expectations!",
        "Wonderful product! It's everything I hoped for and more.",
        "Fantastic! I'm thrilled with this purchase.",
        "This movie was incredible! The acting was superb and the story was engaging.",
        "I had a wonderful time at the restaurant. The food was delicious!",
        "The book is fantastic! I couldn't put it down.",
        "Great job on the presentation! It was very informative.",
        "I'm so happy with my new phone. It works flawlessly!",
        "The weather is beautiful today! Perfect for a picnic.",
        "This software is excellent! It makes my work so much easier.",
        "I love the new design. It's modern and user-friendly.",
        "Outstanding performance! The team did an amazing job.",
        "This is exactly what I needed. Thank you so much!",
    ]
    
    # Sample negative texts
    negative_texts = [
        "This product is terrible. It broke after just one use!",
        "Very disappointed with the quality. Not worth the money.",
        "Poor customer service. They didn't help me at all.",
        "The item arrived damaged and late. Very frustrating experience.",
        "I regret buying this. It doesn't work as advertised.",
        "Awful quality. I expected much better for this price.",
        "This is the worst purchase I've ever made. Complete waste of money.",
        "Terrible experience. I will never buy from them again.",
        "The product is defective and the return process is complicated.",
        "Very unhappy with this service. Would not recommend.",
        "The movie was boring and poorly made. I want my money back.",
        "The restaurant food was cold and tasteless. Terrible experience.",
        "I didn't like the book at all. The plot was confusing.",
        "The presentation was unclear and hard to follow.",
        "My phone stopped working after a week. Very disappointed!",
        "The weather is awful today. It's raining and cold.",
        "This software is buggy and crashes frequently.",
        "I hate the new design. It's confusing and hard to use.",
        "Poor performance. The team needs to do better.",
        "This doesn't work at all. I'm very frustrated!",
    ]
    
    # Sample neutral texts
    neutral_texts = [
        "The product arrived on time. It seems okay so far.",
        "I received the item as expected. Nothing special.",
        "The service was average. It met my basic requirements.",
        "The product is functional. It does what it's supposed to do.",
        "Standard quality. Nothing exceptional, but acceptable.",
        "The item is fine. I have no major complaints.",
        "It's an average product. Not great, not terrible.",
        "The service was okay. I got what I needed.",
        "The product works as described. No issues so far.",
        "It's a regular item. Nothing to write home about.",
        "The movie was okay. It had some good moments.",
        "The restaurant was fine. The food was decent.",
        "The book is average. It's readable but not exciting.",
        "The presentation was informative. It covered the basics.",
        "The phone works. It's a standard smartphone.",
        "The weather is normal for this time of year.",
        "The software works. It has the features I need.",
        "The design is standard. It's functional.",
        "The performance was adequate. Nothing special.",
        "It works. I can use it for my needs.",
    ]
    
    # Create labels
    labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts) + ['neutral'] * len(neutral_texts)
    texts = positive_texts + negative_texts + neutral_texts
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def main():
    """Main function to generate and save sample data"""
    print("Generating sample dataset...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    df = generate_sample_data()
    
    # Save to CSV
    output_path = 'data/sentiment_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print(f"\nDataset saved to {output_path}")

if __name__ == '__main__':
    main()
