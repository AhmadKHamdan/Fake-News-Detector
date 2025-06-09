# Fake News Detector

> Compare baseline TF-IDF vs. enriched NLP pipelines for fake-news classification in a Streamlit app.

## 📂 Repository Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── pipeline.joblib        # (auto-generated) Serialized pipeline artifacts
└── README.md              # Project README
```

## 📋 Project Description

**Fake News Detector** is an interactive Streamlit web application that demonstrates two different machine-learning approaches for classifying news as real or fake:

1. **Baseline Pipeline**: TF-IDF on basic lemmatized text, trained with Decision Tree and Logistic Regression.  
2. **Enriched Pipeline**: Combines TF-IDF (word & character n-grams) with a rich set of NLP features (readability scores, POS/NER counts, sentiment, clickbait heuristics, and topic distributions), also using Decision Tree and Logistic Regression.

The app lets users:
- **Train or load** all four models at once  
- **Inspect** dataset size, feature-matrix shapes, performance metrics, and confusion matrices side-by-side  
- **Drill down** into NLP preprocessing steps on a sample article  
- **Experiment** by entering any sentence and seeing how each model tokenizes, lemmatizes, and classifies it  

This hands-on comparison illustrates how adding stylistic and semantic features can boost fake-news detection performance and highlights the trade-offs between simplicity and accuracy.

## 🔧 Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Create & activate** a Python virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download** NLP models & data (one-time):
   ```bash
   python -m nltk.downloader punkt wordnet stopwords vader_lexicon averaged_perceptron_tagger
   python -m spacy download en_core_web_sm
   ```

## 🚀 Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

Use the sidebar to navigate through three pages:
1. **Run Models**: Train or load four classifiers and view metrics & confusion matrices.  
2. **NLP Demo**: Step-by-step preprocessing on a sample article with feature explanations.  
3. **Predict**: Enter custom text to see model predictions and underlying NLP steps.

## ⚙️ How It Works

- **Baseline Pipeline**: TF-IDF on lemmatized text (word n-grams).  
- **Enriched Pipeline**: Adds char n-grams, readability, POS/NER, sentiment, clickbait, and topic features.

## 📝 License

MIT License — feel free to use and adapt!

---

*Happy fake-news hunting!*
