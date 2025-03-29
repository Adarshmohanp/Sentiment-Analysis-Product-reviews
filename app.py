import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from wordcloud import WordCloud
import re
import os
from textblob import TextBlob

app = Flask(__name__)

# Initialize sentiment analyzer
try:
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("Using 3-class sentiment model from Hugging Face")
except Exception as e:
    print(f"Could not load transformers: {e}")
    print("Falling back to TextBlob")
    sentiment_analyzer = None

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
        if sentiment_analyzer is not None:
            result = sentiment_analyzer(text)[0]
            if result['label'] == 'neutral':
                return "NEUTRAL", 0
            elif result['label'] == 'positive':
                return "POSITIVE", result['score']
            else:
                return "NEGATIVE", -result['score']
        else:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0.3:
                return "POSITIVE", polarity
            elif polarity < -0.3:
                return "NEGATIVE", polarity
            else:
                return "NEUTRAL", polarity
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "NEUTRAL", 0

def detect_aspects(text):
    aspects = []
    text = clean_text(text)
    common_aspects = ['battery', 'camera', 'screen', 'performance', 'price', 'quality', 'delivery', 'service']
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
    
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax1, 
                palette={'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'})
    ax1.set_title('Sentiment Distribution (Count)')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Number of Reviews')
    
    colors = ['green', 'gray', 'red']
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
            df['date'] = pd.to_datetime(df['date'])
            
            daily_trends = df.groupby(['date', 'sentiment']).agg({
                'polarity': 'mean'
            }).unstack()
            
            daily_trends.columns = daily_trends.columns.droplevel(0)
            
            all_dates = pd.date_range(df['date'].min(), df['date'].max())
            daily_trends = daily_trends.reindex(all_dates)
            
            plt.switch_backend('Agg')
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = {'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'}
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
        
        date_col = None
        for col in ['date', 'timestamp', 'time', 'created_at', 'review_date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['date'])
        else:
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
    port = int(os.environ.get("PORT", 10000))  # ← Render uses $PORT
    app.run(host='0.0.0.0', port=port)