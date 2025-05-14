import os
from flask import Flask, render_template, request
import pandas as pd
import re
from textblob import TextBlob
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        if 'text' not in df.columns:
            return "CSV must have a 'text' column", 400

        df['cleaned_text'] = df['text'].astype(str).str.lower()
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
        df.to_csv('result.csv', index=False)

        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts()
        pos_count = sentiment_counts.get("Positive", 0)
        neg_count = sentiment_counts.get("Negative", 0)
        neu_count = sentiment_counts.get("Neutral", 0)

        # Plotly bar chart
        fig = go.Figure(data=[
            go.Bar(name='Sentiments',
                   x=['Positive', 'Negative', 'Neutral'],
                   y=[pos_count, neg_count, neu_count],
                   marker_color=['green', 'red', 'gray'])
        ])
        fig.update_layout(title='Sentiment Distribution',
                          xaxis_title='Sentiment',
                          yaxis_title='Count')
        plot_html = pyo.plot(fig, output_type='div')

        return render_template('result.html',
                               table=df.to_html(classes='data', header="true"),
                               pos=pos_count, neg=neg_count, neu=neu_count,
                               plot_html=plot_html)
    except Exception as e:
        return f"Error processing CSV: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
