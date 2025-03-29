import pandas as pd
import random
from datetime import datetime, timedelta
import os

def generate_sample_reviews(output_file="sample_reviews.csv", num_reviews=50):
    positive_reviews = [
        "Great product, works perfectly!",
        "Excellent quality and fast delivery",
        "Very satisfied with my purchase",
        # ... (keep all your positive reviews)
    ]
    
    neutral_reviews = [
        "It's okay, nothing special",
        "Product works as expected",
        # ... (keep all your neutral reviews)
    ]
    
    negative_reviews = [
        "Poor quality, broke after 2 days",
        "Not worth the money",
        # ... (keep all your negative reviews)
    ]
    
    aspects = ['battery', 'camera', 'screen', 'performance', 'price', 'quality', 'delivery', 'service']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    date_list = [start_date + timedelta(days=x) for x in range(num_reviews)]
    
    reviews = []
    for i in range(num_reviews):
        sentiment_choice = random.random()
        if sentiment_choice < 0.5:
            review = random.choice(positive_reviews)
            sentiment = "POSITIVE"
        elif sentiment_choice < 0.8:
            review = random.choice(neutral_reviews)
            sentiment = "NEUTRAL"
        else:
            review = random.choice(negative_reviews)
            sentiment = "NEGATIVE"
        
        aspect = random.choice(aspects)
        review_text = f"{aspect}: {review}"
        
        reviews.append({
            'date': date_list[i].strftime('%Y-%m-%d'),
            'review': review_text,
            'sentiment': sentiment,
            'aspect': aspect
        })
    
    df = pd.DataFrame(reviews)
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    output_path = os.path.join('uploads', output_file)
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    generate_sample_reviews()