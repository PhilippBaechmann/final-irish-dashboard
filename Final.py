#!/usr/bin/env python
# coding: utf-8

# ======== Imports ========
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Keep even if not directly used, often affects matplotlib styles
import plotly.express as px
import plotly.graph_objects as go
from bertopic import BERTopic
# Make sure corextopic is installed, import might fail otherwise
try:
    from corextopic import corextopic as ct
except ImportError:
    st.error("Could not import 'corextopic'. Please ensure 'corextopic' (not 'corextopic-py') is in requirements.txt and installed.")
    ct = None # Assign None if import fails
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import base64
import io
from PIL import Image
from wordcloud import WordCloud
import nltk
import re
import warnings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import datetime
import asyncio
# ======== End Imports ========


# --- Streamlit Page Config (MUST BE FIRST ST COMMAND) ---
st.set_page_config(
    page_title="Irish CHGFs Analysis Dashboard",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- End Page Config ---


# --- NLTK Setup ---
@st.cache_resource
def download_nltk_resources():
    """Downloads necessary NLTK data (punkt, stopwords) if not found."""
    import nltk
    needed = {'tokenizers/punkt': 'punkt', 'corpora/stopwords': 'stopwords'}
    downloaded_any = False
    print("Checking NLTK resources...") # Print to console, not app UI
    for resource_path, download_name in needed.items():
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{download_name}' found.")
        except LookupError:
            print(f"Downloading NLTK resource '{download_name}'...")
            try:
                nltk.download(download_name, quiet=True)
                print(f"Successfully downloaded '{download_name}'.")
                downloaded_any = True
            except Exception as e:
                print(f"ERROR: Failed to download NLTK resource '{download_name}': {str(e)}")
    return downloaded_any

print("Attempting initial NLTK resource download...")
NLTK_DOWNLOAD_ATTEMPTED = download_nltk_resources()
print("Initial NLTK resource check complete.")

def safe_tokenize(text):
    """Safely tokenizes text, handling potential NLTK errors."""
    if text is None or not isinstance(text, str): return []
    try:
        nltk.data.find('tokenizers/punkt')
        return nltk.word_tokenize(text)
    except LookupError:
        print("WARN: NLTK 'punkt' still not found during tokenization.")
        return re.findall(r'\b\w+\b', text.lower()) # Fallback
    except Exception as e:
        print(f"NLTK tokenization failed: {str(e)}. Using simple fallback.")
        return re.findall(r'\b\w+\b', text.lower())

def safe_get_stopwords():
    """Safely gets NLTK stopwords, handling potential errors."""
    try:
        nltk.data.find('corpora/stopwords')
        return set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        print("WARN: NLTK 'stopwords' still not found.")
        if not NLTK_DOWNLOAD_ATTEMPTED:
            if download_nltk_resources():
                 try: return set(nltk.corpus.stopwords.words('english'))
                 except Exception: return set()
        return set()
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return set()
# --- End NLTK Setup ---


# --- LLM and Environment Variable Setup ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_GROQ = bool(GROQ_API_KEY)
USE_OPENAI = bool(OPENAI_API_KEY)
RAG_ENABLED = USE_GROQ or USE_OPENAI

llm_classes_imported = True
ChatGroq = None
ChatOpenAI = None
try:
    if USE_GROQ: from langchain_groq import ChatGroq
    if USE_OPENAI: from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"ERROR: Failed to import Langchain LLM classes: {e}")
    llm_classes_imported = False
    RAG_ENABLED = False

warnings.filterwarnings('ignore')
# --- End LLM Setup ---


# --- CSS Styling ---
st.markdown("""
<style>
    /* Global reset */
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-sizing: border-box; }
    /* Backgrounds and Text */
    .main, .block-container, .stApp { background-color: #ffffff !important; color: #333333 !important; }
    .block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }
    p, li, div, label, span, .stMarkdown, .stText { color: #333333 !important; line-height: 1.6; }
    table, th, td { color: #333333 !important; }
    /* Headers */
    h1 { color: #103778 !important; font-weight: 700; padding-bottom: 10px; border-bottom: 3px solid #103778; margin-bottom: 25px; font-size: 2rem;} /* Adjusted size */
    h2 { color: #103778 !important; font-weight: 600; padding-bottom: 8px; border-bottom: 1px solid #e0e0e0; margin-bottom: 20px; margin-top: 35px; font-size: 1.6rem;} /* Adjusted size */
    h3 { color: #103778 !important; font-weight: 600; margin-top: 25px; margin-bottom: 15px; font-size: 1.3rem; }
    h4 { color: #1e56a0 !important; font-weight: 600; margin-top: 15px; margin-bottom: 8px; font-size: 1.1rem; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #e0e0e0; padding: 1rem; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { margin-top: 10px; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #ddd; font-size: 1.2rem;}
    /* Buttons */
    .stButton > button { background-color: #103778 !important; color: white !important; border-radius: 5px; border: none; padding: 0.6rem 1.2rem; font-weight: 500; transition: all 0.2s ease; cursor: pointer; }
    .stButton > button:hover { background-color: #1e56a0 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.15); }
    .stButton > button:active { background-color: #0d2b5c !important; }
    /* Metric Cards */
    .metric-card { background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); padding: 20px; text-align: center; transition: all 0.2s; border-left: 5px solid #103778; margin-bottom: 1rem; height: 140px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 5px 12px rgba(0, 0, 0, 0.12); }
    .metric-card span { font-size: 2rem; font-weight: 600; color: #103778; display: block; margin-bottom: 5px;}
    .metric-card p { font-size: 0.9rem; color: #555 !important; margin-top: 0; }
    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: auto; box-shadow: 0 1px 4px rgba(0,0,0,0.06); border: 1px solid #e0e0e0; } /* Added overflow auto */
    .stDataFrame th { background-color: #f1f5f9 !important; color: #333 !important; font-weight: 600; text-align: left; padding: 12px 10px !important; border-bottom: 2px solid #e0e0e0; position: sticky; top: 0; z-index: 1;} /* Sticky header */
    .stDataFrame td { color: #333 !important; padding: 10px !important; vertical-align: top; border-bottom: 1px solid #f0f0f0;}
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 3px; padding: 0; border-bottom: 2px solid #e0e0e0; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: nowrap; background-color: transparent; border-radius: 6px 6px 0 0; border: none; border-bottom: 2px solid transparent; padding: 10px 18px; color: #555 !important; font-weight: 500; transition: all 0.2s ease; }
    .stTabs [aria-selected="true"] { background-color: transparent !important; color: #103778 !important; font-weight: 600; border-bottom: 2px solid #103778 !important; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #f0f0f0 !important; color: #103778 !important;}
    .stTabs [data-baseweb="tab-panel"] { padding: 25px 5px; border: none; }
    /* Expanders */
    .streamlit-expanderHeader { background-color: #f8fafc; border-radius: 5px; padding: 12px !important; font-weight: 600; color: #333 !important; border: 1px solid #e0e0e0; }
    .streamlit-expanderHeader:hover { background-color: #f0f4f8; }
    .streamlit-expanderContent { background-color: #ffffff; border-radius: 0 0 5px 5px; border: 1px solid #e0e0e0; border-top: none; padding: 18px; }
    /* Chat */
    .chat-container { border-radius: 8px; margin-bottom: 12px; padding: 15px 18px; line-height: 1.5; }
    .user-message { background-color: #eef2ff; border-left: 4px solid #4f46e5; }
    .bot-message { background-color: #f0fdf4; border-left: 4px solid #16a34a; }
    .chat-container p { margin: 0; word-wrap: break-word; } /* Added word-wrap */
    .chat-container strong { font-weight: 600; }
    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox > div > div, .stMultiSelect > div > div { border-radius: 5px; border: 1px solid #cbd5e0; transition: border-color 0.2s ease, box-shadow 0.2s ease; }
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox > div > div:focus-within, .stMultiSelect > div > div:focus-within { border-color: #103778; box-shadow: 0 0 0 2px rgba(16, 55, 120, 0.2); }
    .stTextArea textarea { min-height: 100px !important; line-height: 1.5; }
    /* Company Cards */
    .company-card { background-color: white; padding: 18px; border-radius: 8px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #e0e0e0; transition: all 0.2s ease; min-height: 160px; display: flex; flex-direction: column;}
    .company-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.12); transform: translateY(-2px); }
    .company-card h4 { color: #103778; margin-top: 0; margin-bottom: 12px; font-size: 1.15rem;}
    .company-card p { font-size: 0.9rem; margin-bottom: 8px; line-height: 1.5;}
    .company-card strong { font-weight: 600; color: #333; }
    .company-card .description { font-size: 0.85rem; color: #555; max-height: 70px; overflow-y: auto; margin-top: auto; }
</style>
""", unsafe_allow_html=True)
# --- End CSS ---


# --- Display Setup Status ---
nltk_punkt_ok = False
nltk_stopwords_ok = False
try: nltk.data.find('tokenizers/punkt'); nltk_punkt_ok = True
except LookupError: pass
try: nltk.data.find('corpora/stopwords'); nltk_stopwords_ok = True
except LookupError: pass

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
if nltk_punkt_ok: st.sidebar.success("✔ NLTK Punkt Ready")
else: st.sidebar.error("❌ NLTK Punkt Missing")
if nltk_stopwords_ok: st.sidebar.success("✔ NLTK Stopwords Ready")
else: st.sidebar.error("❌ NLTK Stopwords Missing")

if not RAG_ENABLED:
    if not llm_classes_imported: st.sidebar.error("❌ Langchain LLM Import Failed")
    st.sidebar.error("❌ No API Key (Groq/OpenAI)")
    st.sidebar.warning("Competitor Analysis Disabled")
else:
    key_status = []
    if USE_GROQ and GROQ_API_KEY: key_status.append("Groq")
    if USE_OPENAI and OPENAI_API_KEY: key_status.append("OpenAI")
    st.sidebar.success(f"✔ API Key(s) Found: {', '.join(key_status)}")
    st.sidebar.info("Competitor Analysis Enabled")
st.sidebar.markdown("---")
# --- End Setup Status Display ---


# --- Session State Initialization ---
if 'retrieval_chain' not in st.session_state: st.session_state.retrieval_chain = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'competitors_found' not in st.session_state: st.session_state.competitors_found = []
if 'last_analysis' not in st.session_state: st.session_state.last_analysis = None
# --- End Session State ---


# --- Function Definitions ---
@st.cache_data
def load_data(uploaded_file=None):
    """Loads data from uploaded file or default, performs cleaning."""
    df = pd.DataFrame()
    file_to_load = None
    load_source_message = ""
    if uploaded_file:
        file_to_load = uploaded_file
        load_source_message = f"Loading data from uploaded file: {uploaded_file.name}"
    else:
        default_file = 'ireland_cleaned_CHGF.xlsx'
        if os.path.exists(default_file):
            file_to_load = default_file
            load_source_message = f"Loading data from default file: {default_file}"
        else:
            load_source_message = f"Default file {default_file} not found and no file uploaded."
            print(load_source_message)
            return pd.DataFrame() # Return empty if no source

    print(load_source_message) # Print status
    st.sidebar.info(load_source_message) # Show status in sidebar too

    try:
        df = pd.read_excel(file_to_load)
        print(f"Successfully read data. Shape: {df.shape}")

    # --- REVISED AGAIN: Try DROPPING the problematic column ---
        if 'Scaler2021' in df.columns:
            try:
                df = df.drop(columns=['Scaler2021'])
                print("Successfully DROPPED the 'Scaler2021' column.")
                # --- ADD PRINT STATEMENT HERE ---
                print(f"DEBUG load_data: Columns AFTER drop: {df.columns.tolist()}")
            except Exception as drop_e:
                print(f"Error dropping 'Scaler2021' column: {drop_e}")
                # Continue without the column if drop fails for some reason
                pass
        else:
            print("Column 'Scaler2021' not found, nothing to drop.")

        # --- Data Cleaning ---
        original_cols = df.columns.tolist()
        df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True) # Clean names more aggressively
        new_cols = df.columns.tolist()
        if original_cols != new_cols: print(f"Cleaned column names: {original_cols} -> {new_cols}")

        if 'CompanyName' not in df.columns: raise ValueError("Missing required column: 'CompanyName'")

        # Define expected columns and defaults
        expected_cols = {
            'Topic': 'Uncategorized',
            'Description': '',
            'City': 'Unknown',
            'CompanyName': 'Unknown', # Already checked but good practice
            'FoundedYear': None # Use None marker for processing
        }

        for col, default in expected_cols.items():
            if col not in df.columns:
                print(f"Column '{col}' not found. Creating default.")
                df[col] = default
            # Fill NA and ensure string type for text columns
            if default is not None: # Handle FoundedYear separately
                 df[col] = df[col].fillna(default).astype(str)

        # Handle FoundedYear specifically
        df['CompanyAge'] = pd.NA
        if 'FoundedYear' in df.columns:
            df['FoundedYear'] = pd.to_numeric(df['FoundedYear'], errors='coerce')
            initial_rows = len(df)
            df.dropna(subset=['FoundedYear'], inplace=True)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0: print(f"Dropped {dropped_rows} rows due to invalid 'FoundedYear'.")
            if not df.empty:
                 df['FoundedYear'] = df['FoundedYear'].astype(int)
                 current_year = datetime.datetime.now().year
                 df['CompanyAge'] = current_year - df['FoundedYear']
                 df['CompanyAge'] = df['CompanyAge'].apply(lambda x: max(0, x))
                 print("Calculated 'CompanyAge'.")
            else: print("No valid 'FoundedYear' entries left after cleaning.")
        else: print("Column 'FoundedYear' not found.")

        print(f"Data cleaning complete. Final Shape: {df.shape}")
        return df

    except FileNotFoundError:
        st.error(f"Error: File not found at path '{file_to_load}'.")
        return pd.DataFrame()
    except ValueError as ve:
        st.error(f"Data Error: {ve}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading/processing: {str(e)}")
        return pd.DataFrame()

def get_download_link(df, filename, text):
    """Generates an HTML download link for a DataFrame."""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none;background-color:#103778;color:white;padding:8px 12px;border-radius:5px;font-weight:500;display:inline-block;margin-top:10px;">{text} 📥</a>'
    except Exception as e:
        print(f"Error creating download link: {e}")
        return "<span>Error creating download link</span>"


@st.cache_data
def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove non-alpha, strip whitespace."""
    if pd.isna(text) or not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_wordcloud(text, title=None):
    """Generates a WordCloud figure from text."""
    if not text or not text.strip():
         print("Skipping word cloud: Input text is empty.")
         return None
    stopwords_set = safe_get_stopwords()
    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='viridis', stopwords=stopwords_set,
            min_font_size=10, max_font_size=120,
            random_state=42 # Ensure reproducibility
        ).generate(text) # Removed contour for simplicity/potential issues
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        if title: ax.set_title(title, fontsize=16, fontweight='bold', color='#103778')
        plt.tight_layout(pad=0)
        return fig
    except ValueError as ve:
         print(f"ValueError generating word cloud: {ve}")
         st.info("Could not generate word cloud (no words left after filtering?).")
         return None
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        st.error(f"Error generating word cloud: {e}")
        return None

@st.cache_data
def create_city_pie_chart(df_plot):
    """Creates a Plotly Pie chart for city distribution."""
    if 'City' not in df_plot.columns or df_plot['City'].isnull().all(): return None
    city_counts = df_plot['City'].astype(str).value_counts()
    if len(city_counts) > 10:
        top_cities = city_counts.head(10)
        others_count = city_counts[10:].sum()
        if 'Others' in top_cities: others_count += top_cities.pop('Others')
        city_counts_display = pd.concat([top_cities, pd.Series([others_count], index=['Others'])]) if others_count > 0 else top_cities
    else: city_counts_display = city_counts
    if city_counts_display.empty: return None
    fig = px.pie(values=city_counts_display.values, names=city_counts_display.index, title='Company Distribution by City', color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
    fig.update_layout(title_font_size=18, font_size=13, margin=dict(t=50, b=20, l=20, r=20), legend_title_text='Cities')
    fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=11, color='black'), marker=dict(line=dict(color='#ffffff', width=1)))
    return fig

@st.cache_data
def create_company_age_chart(df_plot):
    """Creates a Plotly Bar chart for company age distribution."""
    if 'CompanyAge' not in df_plot.columns or df_plot['CompanyAge'].isnull().all(): return None
    try:
        df_temp = df_plot.copy()
        df_temp['AgeGroup'] = pd.cut(df_temp['CompanyAge'], bins=[0, 3, 5, 10, 15, 20, float('inf')], labels=['0-3', '4-5', '6-10', '11-15', '16-20', '21+'], right=False )
        age_distribution = df_temp['AgeGroup'].value_counts().sort_index()
    except Exception as e: print(f"Error creating age groups: {e}"); return None
    if age_distribution.empty: return None
    fig = px.bar(x=age_distribution.index.astype(str), y=age_distribution.values, color=age_distribution.index.astype(str), labels={'x': 'Company Age (Years)', 'y': 'Number of Companies'}, title='Company Age Distribution', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(title_font_size=18, font_size=13, plot_bgcolor='white', showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'), margin=dict(t=50, b=20, l=20, r=20))
    return fig

@st.cache_data
def create_foundation_year_timeline(df_plot):
    """Creates a Plotly Line chart for company foundations over time."""
    if 'FoundedYear' not in df_plot.columns or df_plot['FoundedYear'].isnull().all(): return None
    yearly_counts = df_plot['FoundedYear'].value_counts().sort_index()
    current_year = datetime.datetime.now().year
    yearly_counts = yearly_counts[(yearly_counts.index > 1950) & (yearly_counts.index <= current_year)]
    if yearly_counts.empty: return None
    fig = px.line(x=yearly_counts.index, y=yearly_counts.values, markers=False, labels={'x': 'Year Founded', 'y': 'Number of Companies Founded'}, title='Company Foundations Over Time')
    fig.add_traces(go.Scatter(x=yearly_counts.index, y=yearly_counts.values, fill='tozeroy', fillcolor='rgba(16, 55, 120, 0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.update_layout(title_font_size=18, font_size=13, plot_bgcolor='white', xaxis=dict(showgrid=False, type='linear'), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'), margin=dict(t=50, b=20, l=20, r=20))
    fig.update_traces(line=dict(color='#103778', width=2.5))
    return fig

@st.cache_resource
def run_bertopic(texts, n_topics=10):
    """Runs BERTopic modeling."""
    if not texts: return None, None
    print(f"Running BERTopic on {len(texts)} descriptions for {n_topics} topics...")
    try:
        model_name = "all-MiniLM-L6-v2"
        try: sentence_model = SentenceTransformer(model_name)
        except Exception as model_e: print(f"Failed to load SentenceTransformer model '{model_name}': {model_e}"); st.error(f"Embedding model failed: {model_e}"); return None, None
        vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1,2))
        topic_model = BERTopic(embedding_model=sentence_model, vectorizer_model=vectorizer, nr_topics=n_topics, calculate_probabilities=True, verbose=False)
        topics, _ = topic_model.fit_transform(texts)
        print("BERTopic fit complete.")
        return topic_model, topics
    except Exception as e: print(f"Error during BERTopic modeling: {e}"); st.error(f"Error during BERTopic modeling: {e}"); return None, None

@st.cache_resource
def run_corex(texts, n_topics=10):
    """Runs CorEx topic modeling."""
    if not texts: return None, None, None
    if ct is None: st.error("corextopic library not imported successfully."); return None, None, None # Check if import failed
    print(f"Running CorEx on {len(texts)} descriptions for {n_topics} topics...")
    try:
        vectorizer = CountVectorizer(stop_words='english', max_features=5000, token_pattern=r'\b[a-zA-Z]{3,}\b')
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0: st.error("No valid words found after preprocessing for CorEx."); return None, None, None
        words = vectorizer.get_feature_names_out()
        topic_model = ct.Corex(n_hidden=n_topics, seed=42)
        topic_model.fit(X, words=words, docs=texts)
        print(f"CorEx fit complete. TC: {topic_model.tc:.4f}")
        topics = [topic_model.get_topics(topic=i, n_words=10, print_words=False) for i in range(n_topics)]
        return topic_model, topics, words
    except Exception as e: print(f"Error during CorEx modeling: {e}"); st.error(f"Error during CorEx modeling: {e}"); return None, None, None

@st.cache_resource
def setup_rag_for_competitor_analysis(_df_rag):
    """Sets up the RAG chain using FAISS and an LLM."""
    global LLM_INITIALIZED
    LLM_INITIALIZED = False
    if not RAG_ENABLED or _df_rag.empty or 'Description' not in _df_rag.columns or _df_rag['Description'].isnull().all():
        print("Skipping RAG setup: Requirements not met.")
        return None
    print("Setting up RAG system...")
    try:
        documents = []
        for _, row in _df_rag[_df_rag['Description'].str.strip().astype(bool)].iterrows():
            content = f"CompanyName: {row.get('CompanyName', 'N/A')}\nDescription: {row.get('Description', '')}\n"
            for col in ['Topic', 'City', 'FoundedYear']: # Use cleaned column names
                if col in _df_rag.columns and pd.notna(row.get(col)):
                    key_name = 'Industry/Topic' if col == 'Topic' else col
                    content += f"{key_name}: {row[col]}\n"
            other_cols = [c for c in _df_rag.columns if c not in ['CompanyName', 'Description', 'Topic', 'City', 'FoundedYear', 'CompanyAge', 'AgeGroup'] and pd.notna(row.get(c))] # Adjusted column names
            for col in other_cols: content += f"{col}: {row[col]}\n"
            documents.append(Document(page_content=content.strip(), metadata={"company": row.get('CompanyName', 'N/A')}))
        if not documents: print("No valid documents created for RAG."); return None
        print(f"Created {len(documents)} documents for RAG.")

        model_name = "all-MiniLM-L6-v2"
        print(f"Loading embedding model: {model_name}")
        try: embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        except Exception as emb_e: print(f"Failed to load embedding model: {emb_e}"); st.error(f"Embedding model failed: {emb_e}"); return None

        print("Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("Vector store created.")

        llm = None
        llm_provider = "None"
        if USE_GROQ and GROQ_API_KEY and ChatGroq:
            try: llm = ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY, temperature=0.6); llm_provider = "Groq"; print("Groq LLM Initialized.")
            except Exception as e: print(f"Groq init failed: {e}.")
        if llm is None and USE_OPENAI and OPENAI_API_KEY and ChatOpenAI:
            try: llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.6); llm_provider = "OpenAI"; print("OpenAI LLM Initialized.")
            except Exception as e: print(f"OpenAI init failed: {e}"); st.error(f"OpenAI LLM failed: {e}"); return None
        if llm is None: print("LLM initialization failed."); st.error("LLM could not be initialized."); return None

        LLM_INITIALIZED = True
        print(f"RAG system ready using {llm_provider}.")
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 7}), return_source_documents=True)
    except Exception as e: print(f"Error setting up RAG system: {e}"); st.error(f"Error setting up RAG: {e}"); return None

def find_potential_competitors(company_name, company_details, retrieval_chain):
    """Queries the RAG chain to find competitors."""
    if retrieval_chain is None: return "Competitor analysis system not available.", []
    print(f"Finding competitors for: {company_name}")
    try:
        query = f"""Analyze the provided database of Irish CHGFs to find potential competitors for: "{company_name}". Company Details: {company_details}. Identify the top 3-5 most similar companies based on business description, industry/sector, and target market. For each potential competitor found ONLY in the database: 1. Provide CompanyName, Description, and Industry/Topic. 2. Briefly explain the key similarity making them a competitor. Exclude "{company_name}" itself. Format clearly, starting with a summary. If no strong competitors are found, state that clearly."""
        result = retrieval_chain({"question": query, "chat_history": []})
        llm_answer = result.get("answer", "Analysis could not be generated.")
        print(f"LLM Answer received (length: {len(llm_answer)}).")
        competitors = []
        seen_companies = {company_name.lower()}
        if "source_documents" in result:
            print(f"Processing {len(result['source_documents'])} source documents.")
            for doc in result["source_documents"]:
                company_info = {}
                comp_name_found = None
                for line in doc.page_content.strip().split('\n'):
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        key, value = key.strip(), value.strip()
                        if key == 'Industry/Topic': key = 'IndustryTopic' # Standardize key for easier access
                        elif key == 'Company Name': key = 'CompanyName'
                        company_info[key] = value
                        if key == "CompanyName": comp_name_found = value
                if comp_name_found and comp_name_found.lower() not in seen_companies:
                    if 'CompanyName' in company_info:
                        # Ensure IndustryTopic exists, map if needed
                        if 'IndustryTopic' not in company_info: company_info['IndustryTopic'] = company_info.get('Topic','N/A')
                        competitors.append(company_info)
                        seen_companies.add(comp_name_found.lower())
                        if len(competitors) >= 5: break
            print(f"Extracted {len(competitors)} unique competitors.")
        else: print("No source documents found in RAG result.")
        return llm_answer, competitors
    except Exception as e: print(f"Error finding competitors: {str(e)}"); st.error(f"Error finding competitors: {str(e)}"); return "An error occurred during competitor analysis.", []
# --- End Function Definitions ---


# ======== Main Application Logic ========
def main():
    # --- Page Title ---
    col1_title, col2_title = st.columns([1, 10], gap="small") # Adjusted ratio
    with col1_title: st.markdown("""<div style="background-color: white; padding: 5px; border-radius: 5px; text-align: center; height: 60px; display: flex; align-items: center; justify-content: center;"><div style="display: flex; height: 40px; width: 60px; border: 1px solid #ccc;"><div style="background-color: #169b62; flex: 1;"></div><div style="background-color: white; flex: 1;"></div><div style="background-color: #ff883e; flex: 1;"></div></div></div>""", unsafe_allow_html=True)
    with col2_title: st.title("Irish CHGFs Analysis Dashboard")

    # --- Data Input & Loading ---
    st.sidebar.markdown("## 📂 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Excel File (Optional)", type=['xlsx', 'xls'])
    df = load_data(uploaded_file)

    # --- Check if DataFrame loaded correctly --- ADDED FIX HERE ---
    if df is None:
        st.error("Fatal Error: Data loading returned None. Cannot proceed.")
        st.stop()
    elif df.empty:
        st.warning("No data loaded or found. Please upload a file or ensure 'ireland_cleaned_CHGF.xlsx' is present.")
        # Optionally allow app to proceed with limited functionality or stop
        st.stop()
    else:
        print("DataFrame loaded successfully in main(). Proceeding...")
    # --- End DataFrame Check ---


    # --- Initialize RAG System ---
    if RAG_ENABLED and 'retrieval_chain' not in st.session_state: # Initialize only once
         print("Attempting RAG setup...")
         st.session_state.retrieval_chain = setup_rag_for_competitor_analysis(df)
         # No rerun here, status displayed via sidebar logic


    # --- Sidebar Filters ---
    st.sidebar.markdown("## 📊 Global Filters")
    df_filtered = df.copy() # Start with the validated DataFrame
    # Topic Filter
    if 'Topic' in df_filtered.columns:
        all_topics = sorted([t for t in df_filtered['Topic'].unique() if pd.notna(t) and t != 'Uncategorized'])
        if all_topics:
            selected_topics = st.sidebar.multiselect("Filter by Topic", options=all_topics, default=[])
            if selected_topics: df_filtered = df_filtered[df_filtered['Topic'].isin(selected_topics)]
    # City Filter
    if 'City' in df_filtered.columns:
         all_cities = sorted([c for c in df_filtered['City'].unique() if pd.notna(c) and c != 'Unknown'])
         if all_cities:
            selected_cities = st.sidebar.multiselect("Filter by City", options=all_cities, default=[])
            if selected_cities: df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]
    # Age Filter - Use CompanyAge column name
    if 'CompanyAge' in df_filtered.columns and not df_filtered['CompanyAge'].isnull().all():
         min_age, max_age = int(df_filtered['CompanyAge'].min()), int(df_filtered['CompanyAge'].max())
         if max_age > min_age:
            age_range = st.sidebar.slider("Filter by Company Age", min_age, max_age, (min_age, max_age))
            df_filtered = df_filtered[df_filtered['CompanyAge'].between(age_range[0], age_range[1])]
    # Name Search Filter - Use CompanyName column name
    if 'CompanyName' in df_filtered.columns:
        company_search = st.sidebar.text_input("Search by Company Name")
        if company_search:
            df_filtered = df_filtered[df_filtered['CompanyName'].str.contains(company_search, case=False, na=False)]

    # Display Filtered Count & Download
    st.sidebar.markdown(f"""<div style="background-color: #eef2ff; padding: 10px; border-radius: 5px; margin-top: 15px; text-align: center; font-size: 0.9rem;"><span style="font-weight: 600;">{len(df_filtered)}</span> / {len(df)} companies showing</div>""", unsafe_allow_html=True)
    if not df_filtered.empty: st.sidebar.markdown("<div style='text-align: center; margin-top: 10px;'>"+get_download_link(df_filtered, 'filtered_data.csv', 'Download Filtered')+"</div>", unsafe_allow_html=True)
    else: st.sidebar.info("No companies match filters.")


    # --- Main Content Tabs ---
    st.markdown("""<div style="background-color: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #103778;"><p style="margin: 0; font-size: 1.0rem;">Explore data, analyze topics, and find competitors among Irish CHGFs.</p></div>""", unsafe_allow_html=True)
    tab_titles = ["📊 Dashboard", "🔍 Explorer", "🏷️ Topic Analysis", "🧠 Adv. Modeling", "🥇 Competitor AI"]
    tabs = st.tabs(tab_titles)


    # ======== TAB 0: Dashboard ========
    with tabs[0]:
        st.header("Dashboard Overview")
        col1, col2, col3, col4 = st.columns(4)
        df_display_metrics = df_filtered if not df_filtered.empty else df # Use filtered or full
        num_comps = len(df_display_metrics)
        num_topics = df_display_metrics['Topic'].nunique() if 'Topic' in df_display_metrics else 0
        num_cities = df_display_metrics['City'].nunique() if 'City' in df_display_metrics else 0
        coverage = round((len(df_display_metrics[df_display_metrics['Topic'] != 'Uncategorized']) / num_comps) * 100, 1) if num_comps > 0 and 'Topic' in df_display_metrics else 0

        with col1: st.markdown(f"<div class='metric-card'><span>{num_comps}</span><p>Companies Shown</p></div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-card'><span>{num_topics}</span><p>Unique Topics</p></div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-card'><span>{num_cities}</span><p>Unique Cities</p></div>", unsafe_allow_html=True)
        with col4: st.markdown(f"<div class='metric-card'><span>{coverage}%</span><p>Topic Coverage</p></div>", unsafe_allow_html=True)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
             fig = create_city_pie_chart(df_display_metrics)
             if fig: st.plotly_chart(fig, use_container_width=True)
             else: st.caption("City Distribution unavailable.")
        with chart_col2:
             fig = create_company_age_chart(df_display_metrics)
             if fig: st.plotly_chart(fig, use_container_width=True)
             else: st.caption("Company Age Distribution unavailable.")

        fig = create_foundation_year_timeline(df_display_metrics)
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.caption("Foundation Year Timeline unavailable.")

        st.markdown("---")
        st.subheader("Sample Data Preview (First 10 Rows of Full Dataset)")
        st.dataframe(df.head(10), use_container_width=True, height=300)


    # ======== TAB 1: Company Explorer ========
    with tabs[1]:
        st.header("Company Explorer")
        # --- ADDED None CHECK ---
        if df_filtered is None:
            st.error("Data filtering error occurred.")
        elif df_filtered.empty:
             st.warning("No companies match the current filter criteria set in the sidebar.")
        else:
            col1_exp, col2_exp = st.columns([1, 3])
            with col1_exp:
                st.markdown("#### Display Options")
                sortable_cols = [col for col in ['CompanyName', 'Topic', 'City', 'FoundedYear', 'CompanyAge'] if col in df_filtered.columns] # Use cleaned names
                sort_by = st.selectbox("Sort by", sortable_cols, key="explorer_sort")
                sort_asc = st.radio("Order", ["Ascending", "Descending"], index=0, key="explorer_order") == "Ascending"
                items_per_page = st.slider("Items per page", 5, 50, 10, 5, key="explorer_paginate")

            df_display = df_filtered.sort_values(by=sort_by, ascending=sort_asc, na_position='last') if sort_by in df_filtered else df_filtered

            with col2_exp:
                total_pages = max(1, (len(df_display) + items_per_page - 1) // items_per_page)
                current_page = 1 # Default page
                if total_pages > 1: current_page = st.number_input("Page", 1, total_pages, 1, 1, key="explorer_page_num")
                start_idx, end_idx = (current_page - 1) * items_per_page, min(current_page * items_per_page, len(df_display))
                st.caption(f"Showing companies {start_idx + 1} to {end_idx} of {len(df_display)}")

            st.markdown("---")
            if start_idx >= end_idx: st.info("No companies to display on this page.")
            else:
                for _, row in df_display.iloc[start_idx:end_idx].iterrows():
                    with st.expander(f"{row.get('CompanyName', 'N/A')}", expanded=False):
                        exp_col1, exp_col2 = st.columns([1, 3])
                        with exp_col1:
                            st.markdown(f"**Topic:** {row.get('Topic', 'N/A')}")
                            st.markdown(f"**City:** {row.get('City', 'N/A')}")
                            if pd.notna(row.get('FoundedYear')): st.markdown(f"**Founded:** {int(row['FoundedYear'])}")
                            if pd.notna(row.get('CompanyAge')): st.markdown(f"**Age:** {int(row['CompanyAge'])} years")
                        with exp_col2:
                            st.markdown(f"**Description:**")
                            desc = row.get('Description', '')
                            st.caption(desc if desc else "No description available.")
                            other_data = {k:v for k,v in row.items() if k not in ['CompanyName','Topic','City','FoundedYear','CompanyAge','Description','AgeGroup'] and pd.notna(v)} # Adjusted names
                            if other_data: st.json(other_data, expanded=False)


    # ======== TAB 2: Topic Analysis ========
    with tabs[2]:
        st.header("Topic Analysis")
        if 'Topic' not in df.columns or 'Description' not in df.columns:
            st.warning("Topic analysis requires both 'Topic' and 'Description' columns.")
        else:
            all_topics = sorted([topic for topic in df['Topic'].unique() if pd.notna(topic)])
            if not all_topics: st.warning("No valid topics found.")
            else:
                st.markdown("#### Select Topic to Analyze")
                topic_for_analysis = st.selectbox("", options=all_topics, key="topic_analysis_select_tab2_final")

                combined_description = "" # Defensive init
                topic_companies = pd.DataFrame()

                if topic_for_analysis:
                    topic_companies = df[df['Topic'] == topic_for_analysis].copy()
                    if not topic_companies.empty: combined_description = ' '.join(topic_companies['Description'].fillna('').astype(str))
                    else: combined_description = ""

                    st.markdown(f"""<div style="background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #103778;"><h3 style="color: #103778; margin-top: 0;">Analysis: {topic_for_analysis} ({len(topic_companies)} companies)</h3></div>""", unsafe_allow_html=True)
                    with st.expander("View Companies & Download", expanded=False):
                        if not topic_companies.empty:
                             st.dataframe(topic_companies[['CompanyName', 'Description']], use_container_width=True, height=200) # Use clean names
                             st.markdown(get_download_link(topic_companies, f'topic_{topic_for_analysis}.csv', f'Download Data'), unsafe_allow_html=True)
                        else: st.write("No companies found for this topic.")

                    wc_col, terms_col = st.columns(2)
                    with wc_col:
                        st.subheader("Topic Word Cloud")
                        if combined_description.strip():
                            processed_desc_for_wc = preprocess_text(combined_description)
                            if processed_desc_for_wc:
                                wordcloud_fig = generate_wordcloud(processed_desc_for_wc)
                                if wordcloud_fig: st.pyplot(wordcloud_fig)
                            else: st.caption("Description empty after preprocessing.")
                        else: st.caption("No description data for word cloud.")
                    with terms_col:
                        st.subheader("Most Common Terms")
                        if combined_description.strip():
                            try:
                                stop_words = safe_get_stopwords()
                                tokens = safe_tokenize(preprocess_text(combined_description))
                                words = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]
                                if words:
                                    word_freq = nltk.FreqDist(words)
                                    top_words = pd.DataFrame(word_freq.most_common(15), columns=['Term', 'Frequency'])
                                    fig_common = px.bar(top_words.sort_values('Frequency'), x='Frequency', y='Term', orientation='h', height=400, title=f'Top Terms', color_discrete_sequence=px.colors.qualitative.Pastel)
                                    fig_common.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=10,r=10,t=30,b=10), font_size=12, plot_bgcolor='white', title_font_size=16)
                                    st.plotly_chart(fig_common, use_container_width=True)
                                else: st.caption("No common terms found after filtering.")
                            except Exception as e: st.error(f"Error analyzing terms: {e}")
                        else: st.caption("No description data for term analysis.")


    # ======== TAB 3: Advanced Topic Modeling ========
    with tabs[3]:
        st.header("Advanced Topic Modeling")
        if 'Description' not in df.columns: st.warning("Advanced Topic Modeling requires the 'Description' column.")
        elif ct is None and modeling_method == "CorEx": st.warning("CorEx library failed to import, cannot run CorEx modeling.") # Check if ct loaded
        else:
            descriptions = df['Description'].dropna().astype(str).tolist()
            descriptions = [d for d in descriptions if len(d.split()) > 5]
            if not descriptions: st.warning("Not enough valid descriptions found for topic modeling.")
            else:
                st.markdown("<p>Discover underlying themes using BERTopic or CorEx.</p>", unsafe_allow_html=True)
                modeling_options = ["BERTopic"]
                if ct is not None: modeling_options.append("CorEx") # Only show CorEx if import worked
                modeling_method = st.radio("Select Method", modeling_options, key="adv_model_method", horizontal=True)

                num_topics = st.slider("Number of Topics", 5, 30, 10, 1, key="adv_num_topics")
                if st.button(f"Run {modeling_method}", key=f"run_adv_{modeling_method}"):
                     with st.spinner(f"Running {modeling_method}..."):
                         if modeling_method == "BERTopic":
                             topic_model, topics = run_bertopic(descriptions, num_topics)
                             if topic_model and topics is not None:
                                 st.success("BERTopic modeling complete!")
                                 topic_info = topic_model.get_topic_info()
                                 st.dataframe(topic_info, use_container_width=True, height=300)
                                 st.caption("Topic -1 represents outliers.")
                         elif modeling_method == "CorEx" and ct is not None: # Ensure ct exists
                             topic_model, topics, words = run_corex(descriptions, num_topics)
                             if topic_model and topics:
                                 st.success("CorEx modeling complete!")
                                 st.write(f"Total Correlation (TC): {topic_model.tc:.3f}")
                                 for i, topic_w in enumerate(topics):
                                     words_str = ", ".join([f"{w}({s:.2f})" for w, s in topic_w[:8]]) # Show top 8 with scores
                                     st.markdown(f"**Topic {i}:** {words_str}...")


    # ======== TAB 4: Competitor Analysis ========
    with tabs[4]:
        st.header("Competitor Analysis AI")
        if not RAG_ENABLED: st.error("Competitor Analysis Disabled (Requires API Key & Setup).")
        elif st.session_state.retrieval_chain is None: st.error("RAG system is not initialized (check setup status in sidebar).")
        else:
            with st.form("competitor_form_tab4"):
                st.markdown("**Enter Your Company Details:**")
                c1f, c2f = st.columns(2)
                with c1f: company_name_input = st.text_input("Your Company Name*", key="ca_comp_name")
                with c2f:
                     # Use cleaned 'Topic' column
                     industry_options = sorted([t for t in df['Topic'].unique() if t not in ['Uncategorized', 'Unknown']])
                     industry_type_input = st.selectbox("Your Industry/Sector*", options=[""] + industry_options, key="ca_industry", index=0)
                company_description_input = st.text_area("Describe your company*", height=120, key="ca_desc")
                submitted = st.form_submit_button("Find Potential Competitors")

            if submitted:
                if not all([company_name_input, industry_type_input, company_description_input]): st.warning("Please fill required (*) fields.")
                else:
                    with st.spinner("Analyzing..."):
                        full_desc_query = f"Industry: {industry_type_input}. Description: {company_description_input}"
                        analysis_text, comps_found = find_potential_competitors(company_name_input, full_desc_query, st.session_state.retrieval_chain)
                        st.session_state.competitors_found = comps_found
                        st.session_state.last_analysis = analysis_text
                        st.rerun() # Rerun to display

            if st.session_state.get('last_analysis'):
                 st.markdown("---")
                 st.markdown("#### AI Analysis Summary")
                 st.markdown(f"<div class='chat-container bot-message'>{st.session_state.last_analysis}</div>", unsafe_allow_html=True)

            if st.session_state.get('competitors_found'):
                 st.markdown("#### Top Matching Companies from Database")
                 num_to_display = min(5, len(st.session_state.competitors_found))
                 cols = st.columns(min(3, num_to_display))
                 for i, comp in enumerate(st.session_state.competitors_found[:num_to_display]):
                     with cols[i % len(cols)]:
                         # Use standardized keys 'CompanyName', 'Description', 'IndustryTopic'
                         st.markdown(f"""<div class="company-card"><h4>{comp.get('CompanyName', '?')}</h4><p><strong>Industry:</strong> {comp.get('IndustryTopic', 'N/A')}</p><p class='description'>{comp.get('Description', 'N/A')}</p></div>""", unsafe_allow_html=True)
                 comp_df = pd.DataFrame(st.session_state.competitors_found[:num_to_display])
                 st.markdown(get_download_link(comp_df, 'competitors.csv','Download Details'), unsafe_allow_html=True)
            elif st.session_state.get('last_analysis'): st.info("No specific competitor details extracted.")

            st.markdown("---")
            st.markdown("#### Ask Follow-up Questions")
            chat_history_display = st.container(height=300)
            with chat_history_display:
                 for i, msg in enumerate(st.session_state.chat_history): st.markdown(f"<div class='chat-container {'user-message' if msg['role']=='user' else 'bot-message'}'><p><strong>{'You' if msg['role']=='user' else 'AI'}:</strong> {msg['content']}</p></div>", unsafe_allow_html=True)

            with st.form(key='chat_form_followup', clear_on_submit=True):
                 chat_input = st.text_area("Your question:", key="ca_chat_input", height=80, label_visibility="collapsed")
                 send_button = st.form_submit_button("Send Question")

            if send_button and chat_input:
                 st.session_state.chat_history.append({"role": "user", "content": chat_input})
                 context_for_llm = "Analysis Context:\n" + st.session_state.get('last_analysis', "No prior analysis.")[:1000] + "\nUser Question:"
                 formatted_hist = [(msg["content"], resp["content"]) for msg, resp in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])]
                 with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.retrieval_chain({"question": context_for_llm + chat_input, "chat_history": formatted_hist})
                        response = result.get("answer", "Sorry, I couldn't process that.")
                    except Exception as e: response = f"An error occurred: {str(e)}"
                 st.session_state.chat_history.append({"role": "assistant", "content": response})
                 st.rerun()

            if st.button("Clear Chat", key="ca_clear_chat"):
                st.session_state.chat_history = []
                st.session_state.competitors_found = []
                st.session_state.last_analysis = None
                st.rerun()


# --- App Entry Point ---
if __name__ == "__main__":
    # Basic asyncio check
    try: asyncio.get_running_loop()
    except RuntimeError: asyncio.set_event_loop(asyncio.new_event_loop())
    main()