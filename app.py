import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
from transformers import pipeline
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
import os
import random

app = Flask(__name__)

# Initialize sentiment analyzer
try:
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Using Hugging Face transformers")
except Exception as e:
    print(f"Could not load transformers: {e}")
    from textblob import TextBlob
    print("Falling back to TextBlob")

# Database setup
def init_db():
    with sqlite3.connect('reviews.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reviews
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT,
                      sentiment TEXT,
                      polarity REAL,
                      timestamp TEXT,
                      aspect TEXT)''')
        conn.commit()

init_db()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def analyze_sentiment(text):
    try:
        if 'sentiment_analyzer' in globals():
            result = sentiment_analyzer(text)[0]
            if 0.4 <= result['score'] <= 0.6:
                return "NEUTRAL", result['score']
            return result['label'], result['score'] if result['label'] == 'POSITIVE' else -result['score']
        else:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0.15:
                return "POSITIVE", polarity
            elif polarity < -0.15:
                return "NEGATIVE", polarity
            else:
                return "NEUTRAL", polarity
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "NEUTRAL", 0

def detect_aspects(text):
    aspects = []
    text = clean_text(text)
    common_aspects = ['battery', 'camera', 'screen', 'performance', 'price', 'quality', 'delivery']
    for aspect in common_aspects:
        if aspect in text:
            aspects.append(aspect)
    return aspects[0] if aspects else 'general'

def generate_sentiment_distribution(results):
    df = pd.DataFrame(results)
    counts = df['sentiment'].value_counts()
    sentiment_counts = {
        "POSITIVE": 0,
        "NEUTRAL": 0,
        "NEGATIVE": 0
    }
    sentiment_counts.update(counts.to_dict())
    return {
        "positive": sentiment_counts["POSITIVE"],
        "neutral": sentiment_counts["NEUTRAL"],
        "negative": sentiment_counts["NEGATIVE"],
        "total": len(results)
    }

def generate_sentiment_chart(results):
    df = pd.DataFrame(results)
    sentiment_counts = df['sentiment'].value_counts()
    
    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax1, 
                palette={'POSITIVE': 'green', 'NEUTRAL': 'orange', 'NEGATIVE': 'red'})
    ax1.set_title('Sentiment Distribution (Count)')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Number of Reviews')
    
    # Pie chart
    colors = ['green', 'orange', 'red']
    ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Sentiment Distribution (%)')
    ax2.axis('equal')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_wordclouds(results):
    df = pd.DataFrame(results)
    positive_text = ' '.join(df[df['sentiment'] == 'POSITIVE']['text'])
    neutral_text = ' '.join(df[df['sentiment'] == 'NEUTRAL']['text'])
    negative_text = ' '.join(df[df['sentiment'] == 'NEGATIVE']['text'])
    
    def create_wordcloud(text):
        if not text:
            return None
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        buf = BytesIO()
        wordcloud.to_image().save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        "positive": create_wordcloud(positive_text),
        "neutral": create_wordcloud(neutral_text),
        "negative": create_wordcloud(negative_text)
    }

def generate_aspect_analysis(results):
    df = pd.DataFrame(results)
    if 'aspect' not in df.columns:
        return {}
    
    aspect_sentiment = df.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    
    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        if sentiment not in aspect_sentiment.columns:
            aspect_sentiment[sentiment] = 0
    
    return aspect_sentiment.to_dict('index')

def generate_trend_analysis():
    try:
        with sqlite3.connect('reviews.db') as conn:
            # Get data with proper date handling
            df = pd.read_sql("""
                SELECT 
                    date(timestamp) as date,
                    polarity,
                    sentiment 
                FROM reviews 
                WHERE timestamp IS NOT NULL
                ORDER BY date
            """, conn)
        
        if not df.empty:
            # Convert date string to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Group by date and sentiment
            daily_trends = df.groupby(['date', 'sentiment']).agg({
                'polarity': 'mean'
            }).unstack()
            
            # Flatten the multi-index columns
            daily_trends.columns = daily_trends.columns.droplevel(0)
            
            # Fill missing dates with NaN
            all_dates = pd.date_range(df['date'].min(), df['date'].max())
            daily_trends = daily_trends.reindex(all_dates)
            
            # Plotting
            plt.switch_backend('Agg')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = {'POSITIVE': 'green', 'NEUTRAL': 'orange', 'NEGATIVE': 'red'}
            for sentiment in daily_trends.columns:
                if sentiment in colors:
                    daily_trends[sentiment].plot(
                        ax=ax,
                        label=sentiment,
                        color=colors[sentiment],
                        marker='o',
                        linestyle='-',
                        alpha=0.7
                    )
            
            ax.set_title('Daily Sentiment Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Average Sentiment Score')
            ax.legend(title='Sentiment')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        
        print("No data available for trend analysis")
        return ""
    except Exception as e:
        print(f"Error in generate_trend_analysis: {str(e)}")
        return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        if 'review' not in df.columns:
            os.remove(filepath)
            return jsonify({"error": "CSV must contain a 'review' column"}), 400
        
        # Handle date column
        date_col = None
        for col in ['date', 'timestamp', 'time', 'created_at', 'review_date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
        else:
            # If no date column, use current time for all reviews
            df['date'] = datetime.now()
        
        if df.empty:
            os.remove(filepath)
            return jsonify({"error": "No valid data found in the file"}), 400
        
        results = []
        with sqlite3.connect('reviews.db') as conn:
            c = conn.cursor()
            for _, row in df.iterrows():
                review_text = str(row['review'])
                sentiment, score = analyze_sentiment(review_text)
                aspect = detect_aspects(review_text)
                review_date = row['date']
                
                # Store date in ISO format
                timestamp = review_date.isoformat()
                
                c.execute("""INSERT INTO reviews 
                            (text, sentiment, polarity, timestamp, aspect) 
                            VALUES (?, ?, ?, ?, ?)""",
                          (review_text, sentiment, score, timestamp, aspect))
                
                results.append({
                    "text": review_text,
                    "sentiment": sentiment,
                    "score": score,
                    "aspect": aspect,
                    "date": review_date.strftime('%Y-%m-%d')
                })
            conn.commit()
        
        sentiment_dist = generate_sentiment_distribution(results)
        sentiment_chart = generate_sentiment_chart(results)
        wordclouds = generate_wordclouds(results)
        aspect_analysis = generate_aspect_analysis(results)
        trend_img = generate_trend_analysis()
        
        os.remove(filepath)
        
        return render_template('results.html', 
                             results=results[:10], 
                             sentiment_dist=sentiment_dist,
                             sentiment_chart=sentiment_chart,
                             wordclouds=wordclouds, 
                             aspect_analysis=aspect_analysis, 
                             trend_img=trend_img)
    
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

@app.route('/generate-sample', methods=['GET'])
def generate_sample():
    try:
        # Generate sample data with realistic dates
        positive_reviews = [
            "Great product, works perfectly!",
            "Excellent quality and fast delivery",
            "Very satisfied with my purchase"
        ]
        neutral_reviews = [
            "It's okay, nothing special",
            "Product works as expected"
        ]
        negative_reviews = [
            "Poor quality, broke after 2 days",
            "Not worth the money"
        ]
        aspects = ['battery', 'camera', 'screen', 'performance']
        
        # Generate dates spread over 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_list = [start_date + timedelta(days=x) for x in range(0, 90, 5)]  # Every 5 days
        
        reviews = []
        for i, date in enumerate(date_list[:20]):  # Generate 20 reviews
            sentiment_choice = random.random()
            if sentiment_choice < 0.6:
                review = random.choice(positive_reviews)
                sentiment = "POSITIVE"
            elif sentiment_choice < 0.8:
                review = random.choice(neutral_reviews)
                sentiment = "NEUTRAL"
            else:
                review = random.choice(negative_reviews)
                sentiment = "NEGATIVE"
            
            aspect = random.choice(aspects)
            review = f"{aspect}: {review}"
            
            reviews.append({
                'date': date.strftime('%Y-%m-%d'),
                'review': review,
                'sentiment': sentiment,
                'aspect': aspect
            })
        
        # Create CSV
        df = pd.DataFrame(reviews)
        csv_path = os.path.join('uploads', 'sample_reviews_with_dates.csv')
        df.to_csv(csv_path, index=False)
        
        return jsonify({
            "message": "Sample CSV with dates generated successfully",
            "path": csv_path,
            "download_url": f"/download-sample?path={csv_path}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-sample')
def download_sample():
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)

@app.route('/debug-db')
def debug_db():
    with sqlite3.connect('reviews.db') as conn:
        data = pd.read_sql("SELECT timestamp, sentiment, polarity FROM reviews LIMIT 10", conn)
        stats = pd.read_sql("""
            SELECT 
                COUNT(*) as total_reviews,
                COUNT(DISTINCT date(timestamp)) as unique_days,
                MIN(date(timestamp)) as first_date,
                MAX(date(timestamp)) as last_date
            FROM reviews
        """, conn)
    
    return jsonify({
        "sample_data": data.to_dict('records'),
        "stats": stats.to_dict('records')[0]
    })

if __name__ == '__main__':
    app.run(debug=True)