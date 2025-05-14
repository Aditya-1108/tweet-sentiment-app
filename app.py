import os
from flask import Flask, render_template, request
import pandas as pd
import re
from textblob import TextBlob
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Set upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Home route for the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    
    if not file:
        return "No file uploaded", 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Save the file to the uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Read CSV into pandas
        df = pd.read_csv(filepath)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            return "CSV must have a 'text' column", 400

        # Clean the text and perform sentiment analysis
        df['cleaned_text'] = df['text'].astype(str).str.lower()
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

        # Count the sentiments
        sentiment_counts = df['sentiment'].value_counts()

        # Create pie chart with Plotly
        fig = px.pie(values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution')
        pie_chart_html = pio.to_html(fig, full_html=False)

        # Pass data to result page
        return render_template('result.html', table=df.to_html(classes='data', header="true"), 
                               sentiment_counts=sentiment_counts.to_dict(), pie_chart_html=pie_chart_html)

    except Exception as e:
        return f"Error processing CSV: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
