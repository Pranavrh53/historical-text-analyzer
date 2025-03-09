Historical Text Analyzer



📌 Overview

The Historical Text Analyzer is a Flask-based web application that enables users to analyze historical texts. It provides sentiment analysis, character network visualization, geospatial mapping, translation, timeline visualization, and more.

✨ Features

Sentiment Analysis: Detects positive, negative, or neutral sentiments in the text.

Character Network: Identifies key figures and maps relationships.

Geospatial Mapping: Extracts and plots historical locations using Folium.

Word Cloud: Generates a word cloud representation of common terms.

Emotion Analysis: Identifies primary emotions such as joy, anger, and fear.

Timeline Visualization: Extracts dates and creates an interactive historical timeline.

Text Translation: Supports multilingual translation using Google Translator API.

Speech-to-Text: Transcribes audio files using AssemblyAI.

🚀 Installation

Prerequisites

Python 3.x

Git

Virtual Environment (recommended)

1. Clone the Repository

git clone https://github.com/Pranavrh53/historical-text-analyzer.git
cd historical-text-analyzer

2. Set Up Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up API Keys

Create a .env file in the root directory and add your AssemblyAI API key:

ASSEMBLYAI_API_KEY=your_api_key_here

5. Run the Flask App

python app.py

Then open http://127.0.0.1:5000/ in your browser.

📂 Project Structure

📦 historical-text-analyzer
├── 📄 app.py               # Flask backend logic
├── 📄 index.html           # Frontend HTML
├── 📄 style.css            # Styling for the UI
├── 📂 static               # Static assets
├── 📂 templates            # HTML templates
├── 📄 requirements.txt      # Python dependencies
├── 📄 .gitignore           # Files to exclude from Git
└── 📄 README.md            # Project documentation

🛠️ Technologies Used

Backend: Flask (Python)

Frontend: HTML, CSS, JavaScript (Tailwind CSS, Plotly)

Data Processing: NLTK, TextBlob, WordCloud, NRCLex

Geospatial Analysis: Geopy, Folium

Speech-to-Text: AssemblyAI API

📝 Future Improvements

✅ Support for more historical languages

✅ Improved entity recognition for better historical event tracking

⏳ Integrate a database for saving analysis history

⏳ Add a REST API for external usage
