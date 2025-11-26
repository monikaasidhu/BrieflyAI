import os
import json
import pickle
import logging
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

#CONFIGURATION 
CATEGORIES = ['sci.med', 'comp.graphics', 'talk.politics.misc', 'rec.autos']

BUSINESS_LABELS = {
    'sci.med': 'Medical Triage',
    'comp.graphics': 'IT Support',
    'talk.politics.misc': 'Legal & Compliance',
    'rec.autos': 'Logistics'
}

#PATH SETUP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_CACHE = os.path.join(BASE_DIR, "data_cache")

def create_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"✓ Directories verified")

def clean_document(text):
    """
    CRITICAL: Removes headers so the model doesn't cheat.
    If we leave 'Newsgroups: sci.med', the model just reads that and wins.
    We want it to read the actual paragraph content.
    """
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Skip metadata headers
        if any(line.startswith(prefix) for prefix in 
              ['From:', 'Subject:', 'Organization:', 'Lines:', 'NNTP-Posting-Host:', 'Newsgroups:', 'Path:', 'Xref:']):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()

def load_and_prepare_data():
    """Step 1: Load Data and Split by Document"""
    logger.info("\n[1/6] Reading and Cleaning Data...")
    
    all_texts = []
    all_labels = []
    
    for label_idx, category_name in enumerate(CATEGORIES):
        filename = f"{category_name}.txt"
        file_path = os.path.join(DATA_CACHE, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"❌ Missing file: {filename}")
            return None, None, None, None

        try:
            logger.info(f"   ...Processing {filename}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
                raw_docs = raw_content.split('From: ')
                
                valid_docs = []
                for doc in raw_docs:
                    cleaned_text = clean_document(doc)
                    if len(cleaned_text) > 50:
                        valid_docs.append(cleaned_text)
                
                logger.info(f"      -> extracted {len(valid_docs)} clean documents")
                
                all_texts.extend(valid_docs)
                all_labels.extend([label_idx] * len(valid_docs))
                
        except Exception as e:
            logger.error(f"❌ Error reading {filename}: {e}")
            return None, None, None, None

    logger.info(f"✓ Total training documents: {len(all_texts)}")

    if len(all_texts) == 0:
        logger.error("❌ No data found. Your text files might not use 'From:' as separators.")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def create_ml_pipeline():
    """Step 2: Pipeline Creation (Strict Regularization)"""
    logger.info("\n[2/6] Building ML Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            max_features=5000, 
            # min_df=5: Ignore words that appear in less than 5 documents (Fixes "Mugging up" rare IDs)
            min_df=5, 
            # max_df=0.5: Ignore words that appear in >50% of documents (Too common)
            max_df=0.5 
        )),
        # alpha=0.1: Smoothing to handle unseen words gracefully
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    """Step 3: Model Training"""
    logger.info("\n[3/6] Training classifier...")
    pipeline.fit(X_train, y_train)
    logger.info("✓ Model training complete")
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Step 4: Model Evaluation"""
    logger.info("\n[4/6] Evaluating model performance...")
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy

def save_model_with_versioning(pipeline, accuracy):
    """Step 5 & 6: MLOps Versioning"""
    logger.info("\n[5/6] Saving artifacts...")
    
    try:
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(MODELS_DIR, f"classifier_v{version_id}.pkl")
        metadata_path = os.path.join(MODELS_DIR, f"metadata_v{version_id}.json")
        config_path = os.path.join(MODELS_DIR, "config.json")

        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        metadata = {
            'version_id': version_id,
            'accuracy': float(accuracy),
            'algorithm': 'MultinomialNB + TF-IDF (Strict)',
            # CRITICAL: Save the exact label order
            'class_labels': [BUSINESS_LABELS[cat] for cat in CATEGORIES],
            'training_date': datetime.now().isoformat(),
            'categories_mapping': BUSINESS_LABELS
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(config_path, 'w') as f:
            json.dump({'latest_version': version_id}, f, indent=2)
            
        logger.info(f"✓ Success! Model Version: v{version_id}")
        return version_id

    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        return None

def main():
    logger.info("="*60)
    logger.info("BrieflyAI - Strict Training Pipeline")
    logger.info("="*60)
    
    create_directories()
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    if X_train:
        pipeline = create_ml_pipeline()
        pipeline = train_model(pipeline, X_train, y_train)
        accuracy = evaluate_model(pipeline, X_test, y_test)
        version_id = save_model_with_versioning(pipeline, accuracy)
        
        if version_id:
            logger.info("\n[6/6] COMPLETE. Now run 'python -m streamlit run app.py'")
            
    logger.info("="*60)

if __name__ == "__main__":

    main()
