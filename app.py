from flask import Flask, render_template, request
import re
from collections import Counter
from nrclex import NRCLex
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import io
import base64
import emoji
import matplotlib
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import nltk
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import folium
from folium import IFrame
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import time
from deep_translator import GoogleTranslator
import dateparser
from datetime import datetime
import plotly.graph_objects as go
import json
import requests
from werkzeug.utils import secure_filename
import os
import assemblyai as aai
from flask import Flask, render_template, request, jsonify
from flask import Flask, flash

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")




nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

matplotlib.use('Agg')



app = Flask(__name__)

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def analyze_sentiment(text):
    sentences = text.split('.')
    sentiment_scores = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        sentiment_scores.append(blob.sentiment.polarity)
    
    sentiment_df = pd.DataFrame({
        'polarity': sentiment_scores
    })
    
    if 'polarity' not in sentiment_df.columns or sentiment_df.empty:
        return None, None
    
    if sentiment_df['polarity'].mean() < -0.1:
        sentiment = "negative"
        color = "#ff0000"
    elif sentiment_df['polarity'].mean() > 0.1:
        sentiment = "positive"
        color = "#00ff00"
    else:
        sentiment = "neutral"
        color = "#ffff00"
        
    return sentiment_df, {
        'sentiment': sentiment,
        'color': color,
        'polarity': sentiment_df['polarity'].mean()
    }

def plot_sentiment_meter(sentiment_df):
    if sentiment_df is None:
        return None
    
    # Create figure with horizontal layout
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create the meter background (horizontal semi-circle)
    theta = np.linspace(0, np.pi, 100)
    radius = 1
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Create color gradients
    r = np.zeros(100)
    g = np.zeros(100)
    b = np.zeros(100)
    
    # First third - Red
    third = len(theta) // 3
    r[:third] = 1.0
    g[:third] = np.linspace(0, 1, third)
    
    # Middle third - Yellow
    r[third:2*third] = 1.0
    g[third:2*third] = 1.0
    
    # Last third - Green
    r[2*third:] = np.linspace(1, 0, len(theta) - 2*third)
    g[2*third:] = 1.0
    
    # Combine colors into RGBA array
    colors = np.column_stack((r, g, b, np.full(len(theta), 0.3)))
    
    # Plot the meter background
    ax.scatter(x, y, c=colors, s=500)
    
    # Plot the pointer
    polarity = sentiment_df['polarity'].mean()
    pointer_x = np.cos(polarity * np.pi/2 + np.pi/2)  # Map [-1,1] to [π,0]
    pointer_y = np.sin(polarity * np.pi/2 + np.pi/2)
    
    # Draw pointer line from center
    ax.plot([0, pointer_x], [0, pointer_y], color='black', linewidth=2)
    
    # Add arrow at the end of pointer
    arrow_length = 0.1
    arrow_angle = np.arctan2(pointer_y, pointer_x)
    left_arrow = np.array([
        pointer_x - arrow_length * np.cos(arrow_angle + np.pi/6),
        pointer_y - arrow_length * np.sin(arrow_angle + np.pi/6)
    ])
    right_arrow = np.array([
        pointer_x - arrow_length * np.cos(arrow_angle - np.pi/6),
        pointer_y - arrow_length * np.sin(arrow_angle - np.pi/6)
    ])
    ax.plot([pointer_x, left_arrow[0]], [pointer_y, left_arrow[1]], color='black', linewidth=2)
    ax.plot([pointer_x, right_arrow[0]], [pointer_y, right_arrow[1]], color='black', linewidth=2)
    
    # Add emojis at key positions
    ax.text(-0.9, 0.3, '😊', fontsize=15)  # Negative
    ax.text(0, 1.1, '😐', fontsize=15)     # Neutral
    ax.text(0.9, 0.3, '😠', fontsize=15) # Positive
    
    # Add sentiment score
    score_text = f"Score: {polarity:.2f}"
    ax.text(0, -0.3, score_text, ha='center', va='center', fontsize=12)
    
    # Customize the plot
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect('equal')
    
    # Remove ticks and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout(pad=1.5)
    
    return fig_to_base64(fig)

def extract_characters(text):
    words = re.findall(r'\b[A-Z][a-z]*\b', text)
    return Counter(words).most_common(10)

def create_network(characters):
    G = nx.Graph()
    
    if not characters:
        return G
    
    for char, freq in characters:
        if char and isinstance(char, str):
            G.add_node(char, weight=freq)
    
    if len(G.nodes()) >= 2:
        char_list = list(G.nodes())
        for i in range(len(char_list) - 1):
            G.add_edge(char_list[i], char_list[i + 1])
    
    return G

def plot_network(G):
    if len(G.nodes()) == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, 'No character network to display', 
                horizontalalignment='center', verticalalignment='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    try:
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        
        nx.draw_networkx_nodes(G, pos,
                             node_color="lightblue",
                             node_size=1500,
                             alpha=0.7)
        
        nx.draw_networkx_edges(G, pos,
                             edge_color="gray",
                             width=2,
                             alpha=0.5)
        
        nx.draw_networkx_labels(G, pos,
                              font_size=10,
                              font_weight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
    except Exception as e:
        print(f"Error in plotting network: {str(e)}")
        ax.text(0.5, 0.5, 'Error creating character network', 
                horizontalalignment='center', verticalalignment='center')
    
    return fig_to_base64(fig)

def plot_word_cloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)
    
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    return fig_to_base64(fig)


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches='tight', transparent=True)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def extract_locations(text):
    """Extract location entities from text using NLTK"""
    locations = []
    
    # Tokenize and tag the text
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Extract named entities
    entities = ne_chunk(tagged)
    
    # Look for GPE (Geo-Political Entity) and LOCATION entities
    for entity in entities:
        if hasattr(entity, 'label'):
            if entity.label() in ('GPE', 'LOCATION'):
                location = ' '.join([leaf[0] for leaf in entity.leaves()])
                locations.append(location)
    
    # Additional pattern-based location extraction
    # Common location indicators
    location_indicators = r'\b(in|at|near|from)\s+([A-Z][a-zA-Z\s]+)(?=[\s,\.])'
    matches = re.finditer(location_indicators, text)
    for match in matches:
        potential_location = match.group(2).strip()
        if potential_location not in locations:
            locations.append(potential_location)
    
    return list(set(locations))  # Remove duplicates

def geocode_locations(locations):
    """Convert location names to coordinates using Nominatim"""
    geolocator = Nominatim(user_agent="historical_text_analysis")
    geocoded_locations = []
    
    for loc in locations:
        try:
            # Add delay to respect Nominatim's usage limits
            time.sleep(1)
            location = geolocator.geocode(loc)
            if location:
                geocoded_locations.append({
                    'name': loc,
                    'lat': location.latitude,
                    'lon': location.longitude,
                    'address': location.address
                })
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Error geocoding {loc}: {str(e)}")
            continue
    
    return geocoded_locations

def create_map(locations):
    """Create a Folium map with markers for each location"""
    if not locations:
        return None
    
    # Calculate center of the map
    center_lat = sum(loc['lat'] for loc in locations) / len(locations)
    center_lon = sum(loc['lon'] for loc in locations) / len(locations)
    
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Add markers for each location
    for loc in locations:
        popup_content = f"""
            <div style='width: 200px'>
                <h4>{loc['name']}</h4>
                <p>{loc['address']}</p>
                <a href="https://www.google.com/maps?q={loc['lat']},{loc['lon']}" 
                   target="_blank" style="background-color: #4CAF50; color: white; 
                   padding: 5px 10px; text-decoration: none; border-radius: 4px; 
                   display: inline-block; margin-top: 5px;">
                   View on Google Maps
                </a>
            </div>
        """
        iframe = IFrame(html=popup_content, width=220, height=150)
        popup = folium.Popup(iframe)
        
        folium.Marker(
            [loc['lat'], loc['lon']],
            popup=popup,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    # Return the HTML representation of the map
    return m._repr_html_()

# Add this function near your other utility functions
def translate_text(text, dest_language):
    """
    Translate text to the specified language using deep_translator,
    handling the 5000 character limit by splitting text intelligently.
    
    Args:
        text (str): The text to translate
        dest_language (str): The target language code (e.g., 'es', 'fr', 'de')
    
    Returns:
        str: Translated text or error message
    """
    if not text or not dest_language:
        return "No text or language provided"
    
    # Language code mapping dictionary
    language_mapping = {
        'zh-cn': 'zh-CN',
        'zh-tw': 'zh-TW'
    }
    
    # Convert language code if needed
    dest_language = language_mapping.get(dest_language.lower(), dest_language)
    
    def split_text(text, limit=4500):  # Using 4500 to leave some margin
        """Split text into chunks, trying to break at sentence boundaries"""
        if len(text) <= limit:
            return [text]
        
        chunks = []
        current_chunk = ""
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= limit:  # +2 for '. '
                current_chunk += sentence + '. '
            else:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If sentence itself is longer than limit, split by words
                if len(sentence) > limit:
                    words = sentence.split()
                    current_chunk = ""
                    
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= limit:
                            current_chunk += word + ' '
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word + ' '
                else:
                    current_chunk = sentence + '. '
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    try:
        # Split text into manageable chunks
        chunks = split_text(text)
        translated_chunks = []
        
        # Translate each chunk
        for chunk in chunks:
            try:
                translator = GoogleTranslator(source='auto', target=dest_language)
                translated_chunk = translator.translate(chunk)
                if translated_chunk:
                    translated_chunks.append(translated_chunk)
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.5)
            except Exception as chunk_error:
                print(f"Error translating chunk: {str(chunk_error)}")
                translated_chunks.append(f"[Translation error: {str(chunk_error)}]")
        
        # Join the translated chunks
        final_translation = ' '.join(translated_chunks)
        
        if not final_translation:
            return "Translation returned empty result"
        
        return final_translation
            
    except Exception as e:
        print(f"Translation error details: {str(e)}")
        if "wrong language code" in str(e).lower():
            return f"Invalid language code: {dest_language}"
        elif "blocked" in str(e).lower():
            return "Translation service temporarily unavailable. Please try again later."
        else:
            return f"Translation failed: {str(e)}"



from textblob import TextBlob

def analyze_emotions(text):
    # Use NRCLex to extract emotions from the text
    emotion_analysis = NRCLex(text)
    raw_emotions = emotion_analysis.raw_emotion_scores

    # Filter for only the six primary emotions
    emotions = {emotion: raw_emotions.get(emotion, 0) for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust"]}

    # Normalize emotion values to percentages
    total = sum(emotions.values())
    if total > 0:
        emotions = {key: (value / total) * 100 for key, value in emotions.items()}
    else:
        emotions = {key: 0 for key in emotions}  # Handle cases with no emotions detected

    return emotions



def plot_emotion_pie_chart(emotions):
    labels = [key.capitalize() for key in emotions.keys()]
    sizes = list(emotions.values())  # Percentages for each emotion
    colors = ['#FFD700', '#1E90FF', '#FF4500', '#8B0000', '#32CD32', '#FF69B4']
    explode = [0.05 if size == max(sizes) else 0 for size in sizes]  # Highlight the largest slice

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted figure size for clarity

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, shadow=True, textprops={'fontsize': 12}
    )

    # Add styling to percentage labels
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # Add a title for the chart
    ax.set_title("Emotion Distribution", fontsize=16, fontweight="bold", color="#333")

    # Ensure the chart is a perfect circle
    ax.axis('equal')

    # Add a legend for better clarity
    ax.legend(wedges, labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)

    return fig_to_base64(fig)


def extract_dates_and_events(text):
    """Extract dates and associated events from text"""
    sentences = sent_tokenize(text)
    timeline_data = []
    
    for sentence in sentences:
        # Look for dates in the sentence
        potential_dates = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})[,\s]+\d{4}\b|\b\d{4}\b', sentence, re.IGNORECASE)
        
        for date_str in potential_dates:
            try:
                # Parse the date string to a datetime object
                parsed_date = dateparser.parse(date_str)
                if parsed_date:
                    # Clean up the event text by removing the date
                    event_text = re.sub(date_str, '', sentence).strip()
                    event_text = re.sub(r'^[,\s]+|[,\s]+$', '', event_text)
                    
                    # Convert datetime to string format for JSON serialization
                    date_str = parsed_date.strftime('%Y-%m-%d')
                    
                    timeline_data.append({
                        'date': date_str,
                        'event': event_text,
                        'year': parsed_date.year
                    })
            except Exception as e:
                print(f"Error parsing date {date_str}: {str(e)}")
                continue
    
    # Sort by date
    timeline_data.sort(key=lambda x: dateparser.parse(x['date']))
    return timeline_data

def create_timeline_visualization(timeline_data):
    """Create an interactive timeline visualization using Plotly"""
    if not timeline_data:
        return None
    
    # Extract data
    dates = [item['date'] for item in timeline_data]
    events = [item['event'] for item in timeline_data]
    years = [item['year'] for item in timeline_data]
    
    # Create figure data
    trace = {
        'type': 'scatter',
        'x': dates,
        'y': [0] * len(dates),
        'mode': 'markers+text',
        'marker': {
            'size': 15,
            'color': '#1E90FF',
            'symbol': 'circle'
        },
        'text': years,
        'textposition': 'top center',
        'hovertext': [f"{date}<br>{event}" for date, event in zip(dates, events)],
        'hoverinfo': 'text',
        'name': ''
    }
    
    # Create connecting line
    line_trace = {
        'type': 'scatter',
        'x': dates,
        'y': [0] * len(dates),
        'mode': 'lines',
        'line': {
            'color': '#1E90FF',
            'width': 2
        },
        'hoverinfo': 'skip',
        'name': ''
    }
    
    # Create layout
    layout = {
        'showlegend': False,
        'hovermode': 'closest',
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'yaxis': {
            'showgrid': False,
            'zeroline': False,
            'showticklabels': False,
            'range': [-1, 1]
        },
        'xaxis': {
            'showgrid': True,
            'zeroline': False,
            'type': 'date',
            'title': 'Timeline'
        },
        'margin': {
            'l': 50,
            'r': 50,
            't': 50,
            'b': 50
        },
        'height': 300
    }
    
    return {
        'data': [trace, line_trace],
        'layout': layout
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def analyze_text():
    # Initialize variables
    sample_text = ""
    sentiment_img = None
    network_img = None
    word_cloud_img = None
    sentiment_info = None
    map_html = None
    translated_text = None
    emotion_pie_chart_img = None
    timeline_data = None

    if request.method == "POST":
        sample_text = request.form.get("sample_text", "")
        dest_language = request.form.get("language", "")

        if sample_text:
            if len(sample_text) > 5000:
                flash("Processing long text... This may take a moment.")
            # Translation
            if dest_language:
                
                translated_text = translate_text(sample_text, dest_language)
            
            # Other analyses
            cleaned_text = clean_text(sample_text)
            
            # Sentiment Analysis
            sentiment_df, sentiment_info = analyze_sentiment(cleaned_text)
            if sentiment_df is not None:
                sentiment_img = plot_sentiment_meter(sentiment_df)

            # Character Network
            characters = extract_characters(sample_text)
            G = create_network(characters)
            network_img = plot_network(G)

            # Word Cloud
            word_cloud_img = plot_word_cloud(sample_text)

            # Emotions
            emotions = analyze_emotions(sample_text)
            emotion_pie_chart_img = plot_emotion_pie_chart(emotions)

            # Timeline
            timeline_events = extract_dates_and_events(sample_text)
            if timeline_events:
                timeline_data = create_timeline_visualization(timeline_events)
                timeline_data = json.dumps(timeline_data)

            # Location Map
            try:
                locations = extract_locations(sample_text)
                geocoded_locations = geocode_locations(locations)
                map_html = create_map(geocoded_locations)
            except Exception as e:
                print(f"Error in location analysis: {str(e)}")
                map_html = None

    return render_template("index.html", 
                         sample_text=sample_text,
                         sentiment_img=sentiment_img,
                         network_img=network_img,
                         word_cloud_img=word_cloud_img,
                         sentiment_info=sentiment_info,
                         map_html=map_html,
                         translated_text=translated_text,
                         emotion_pie_chart_img=emotion_pie_chart_img,
                         timeline_data=timeline_data
                         )



aai.settings.api_key = ASSEMBLYAI_API_KEY
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

def transcribe_audio(file_path):
    """Transcribe an audio file using AssemblyAI SDK."""
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(file_path)
        return transcript.text
    except Exception as e:
        return f"Transcription failed: {str(e)}"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    try:
        transcription_text = transcribe_audio(file_path)
        os.remove(file_path)  # Cleanup after transcription
        return jsonify({"success": True, "transcription": transcription_text})
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)  # Cleanup on error
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)