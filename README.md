# BrieflyAI: Intelligent Document Router & Summarizer

An enterprise-grade document intelligence system that automates document triage and generates executive summaries using a hybrid AI architecture.

##  Overview

BrieflyAI addresses the bottleneck of manual document routing in high-volume environments. The system instantly classifies incoming documents into departmental categories and generates concise summaries, reducing review time by up to 80%.

**Key Capabilities:**
- **Multi-class Classification**: Routes documents to Medical, IT, Legal, or Logistics departments
- **Abstractive Summarization**: Generates human-readable executive briefs
- **Production-Ready**: Includes versioned model management and metadata tracking
- **Resource Efficient**: Runs on CPU without GPU requirements

**Hybrid Design:**
- **ML Layer**: TF-IDF + Multinomial Naive Bayes for fast, accurate routing
- **DL Layer**: Pre-trained T5-Small transformer for semantic summarization
- **MLOps**: Automated versioning, metadata tracking, and configuration management

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/monikaasidhu/BrieflyAI.git
cd BrieflyAI

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Download the **20 Newsgroups (By Date)** dataset and extract the following categories into the `data_cache/` directory:

- `sci.med` → Medical Triage
- `comp.graphics` → IT Support
- `talk.politics.misc` → Legal & Compliance
- `rec.autos` → Logistics

Each category should be saved as a `.txt` file (e.g., `sci.med.txt`).

### 3. Train the Classifier

```bash
python train_pipeline.py
```

**Output:**
- Trained model: `models/classifier_v{timestamp}.pkl`
- Metadata: `models/metadata_v{timestamp}.json`
- Config pointer: `models/config.json`

### 4. Launch Application

```bash
streamlit run app.py
```

Access the web interface at `http://localhost:8501`

##  Usage

1. **Input**: Paste document text into the left panel
2. **Process**: Click "Process" button
3. **Output**: View department routing and executive summary in the right panel

##  Technical Details
### Training Pipeline (`train_pipeline.py`)

**Data Preprocessing:**
- Strips metadata headers to prevent data leakage
- Splits documents using delimiter-based parsing
- Filters documents with minimum length threshold

**Model Configuration:**
```python
TfidfVectorizer(
    max_features=5000,    # Top 5000 most important words
    min_df=5,             # Ignore rare terms
    max_df=0.5,           # Ignore overly common terms
    stop_words='english'
)

MultinomialNB(alpha=0.1)  # Laplace smoothing
```

**MLOps Features:**
- Timestamp-based versioning
- JSON metadata with accuracy metrics
- Train-test split (80/20) with fixed random seed

### Application (`app.py`)

**Key Components:**
- `@st.cache_resource` for efficient model loading
- Dynamic version detection from config file
- Error handling for missing dependencies
- Two-column responsive layout

**Performance:**
- Classification: <100ms per document
- Summarization: 2-4 seconds per document (CPU)
- Memory: ~500MB with both models loaded

##  Model Performance

**Test Accuracy:** 85-92% (varies by training run)

**Validation Test:**
```
Input: "I have a throbbing pain in my temples and feel nauseous 
        whenever I see bright light."
Prediction: Medical Triage ✓
```

##  Future Enhancements

- [ ] Add confidence scores to predictions
- [ ] Multi-label classification support
- [ ] REST API for programmatic access
- [ ] A/B testing framework for model comparison
- [ ] Support for additional languages
- [ ] Real-time performance monitoring dashboard
