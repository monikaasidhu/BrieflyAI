"""
BrieflyAI - Application (Final Production Version)
==================================================
Streamlit UI for Document Routing & Summarization.
"""

import streamlit as st
import json
import os
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(page_title="BrieflyAI", page_icon="⚡", layout="wide")

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- LOADERS ---
@st.cache_resource
def load_dl_model():
    """Loads T5 Model with lazy import to prevent startup crashes."""
    try:
        from transformers import pipeline
        return pipeline("summarization", model="t5-small", device=-1)
    except ImportError:
        st.error("Library missing. Run: pip install transformers torch sentencepiece")
        return None
    except Exception as e:
        st.error(f"Deep Learning Model Error: {e}")
        return None

def load_ml_model():
    """Loads the ML Classifier and Metadata."""
    try:
        import pickle
        
        config_path = os.path.join(MODELS_DIR, 'config.json')
        if not os.path.exists(config_path):
            return None, None, "Config missing. Run train_pipeline.py first."
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        version = config['latest_version']
        
        model_path = os.path.join(MODELS_DIR, f'classifier_v{version}.pkl')
        meta_path = os.path.join(MODELS_DIR, f'metadata_v{version}.json')
        
        if not os.path.exists(model_path):
            return None, None, "Model file missing."
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        return model, meta, None
    except Exception as e:
        return None, None, str(e)

# --- UI LOGIC ---
def main():
    st.title("⚡ BrieflyAI: Intelligent Document Router")
    st.markdown("---")

    # Load Systems
    classifier, metadata, error = load_ml_model()
    
    # Show DL loading state clearly
    with st.spinner("Initializing Deep Learning Engine..."):
        summarizer = load_dl_model()

    # Sidebar
    if metadata:
        st.sidebar.success(f"System Online: v{metadata['version_id']}")
        st.sidebar.write(f"**Accuracy:** {metadata['accuracy']*100:.1f}%")
        st.sidebar.markdown("### Active Departments")
        # Display categories in the exact order the model learned them
        if 'class_labels' in metadata:
            for idx, label in enumerate(metadata['class_labels']):
                st.sidebar.text(f"{idx}: {label}")
    else:
        st.sidebar.error("System Offline")
        if error: st.sidebar.warning(error)

    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Input")
        text = st.text_area("Paste document here:", height=300)
        btn = st.button("Process", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Analysis")
        if btn and text:
            if not classifier or not summarizer:
                st.error("Models not loaded.")
            else:
                try:
                    # 1. Routing
                    pred_idx = classifier.predict([text])[0]
                    
                    # Use labels directly from metadata (Correct Order)
                    if 0 <= pred_idx < len(metadata['class_labels']):
                        label = metadata['class_labels'][pred_idx]
                    else:
                        label = "Unknown"
                    
                    st.info(f"📂 **Department:** {label}")
                    
                    # 2. Summarization
                    with st.spinner("Summarizing..."):
                        summary = summarizer(f"summarize: {text}", max_length=300, min_length=30, do_sample=False)
                        st.success("📝 **Executive Brief:**")
                        st.write(summary[0]['summary_text'])
                        
                except Exception as e:
                    st.error(f"Processing Error: {e}")

if __name__ == "__main__":

    main()
