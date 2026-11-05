# src/stylometry_features.py

import os
import spacy # Use spaCy
import re
from collections import Counter
import statistics
import textstat
import json
# Keep NLTK just for sentence tokenization for now, or remove if using doc.sents
from nltk.tokenize import sent_tokenize
# import nltk # No longer needed for POS tagging

# --- Load spaCy Model ---
NLP = None
SPACY_MODEL_NAME = "en_core_web_sm"
try:
    NLP = spacy.load(SPACY_MODEL_NAME)
    print(f"spaCy model '{SPACY_MODEL_NAME}' loaded successfully.")
except OSError:
    print(f"ERROR: spaCy model '{SPACY_MODEL_NAME}' not found.")
    print(f"Please run: python -m spacy download {SPACY_MODEL_NAME}")
    # Depending on requirements, you might want to exit() here or handle gracefully
    # sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading spaCy model: {e}")


# --- Feature Calculation Functions ---

def calculate_sentence_stats(sentences):
    """Calculates average and standard deviation of sentence length (using space splitting)."""
    # Note: Could also get length directly from spaCy spans if sentences were derived from doc.sents
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
    if not sentence_lengths:
        return {'avg_sent_len': 0, 'std_sent_len': 0}
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    std_dev = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
    return {'avg_sent_len': avg_len, 'std_sent_len': std_dev}

def calculate_word_stats(spacy_tokens):
    """Calculates average word length (characters) for alphanumeric spaCy tokens."""
    # spaCy tokens have attributes like .is_alpha, .is_alnum
    # Check if token is alphabetic OR a digit - closer to isalnum logic
    word_lengths = [len(token.text) for token in spacy_tokens if (token.is_alpha or token.is_digit)]
    if not word_lengths:
        return {'avg_word_len': 0}
    avg_len = sum(word_lengths) / len(word_lengths)
    return {'avg_word_len': avg_len}

def calculate_ttr(spacy_tokens):
    """Calculates Type-Token Ratio using spaCy token texts."""
    if not spacy_tokens:
        return {'ttr': 0}
    # Use the .text attribute for the actual word string
    all_token_texts = [token.text.lower() for token in spacy_tokens] # Lowercase for TTR
    num_tokens = len(all_token_texts)
    num_types = len(set(all_token_texts))
    ttr = num_types / num_tokens if num_tokens > 0 else 0
    return {'ttr': ttr}

def calculate_pos_freq_spacy(spacy_tokens, top_n=None):
    """Calculates Part-of-Speech tag frequencies using spaCy token.pos_."""
    if not spacy_tokens:
        return {'pos_freq': {}}
    # Use the simple Universal POS tags (token.pos_)
    tag_counts = Counter(token.pos_ for token in spacy_tokens if not token.is_space)
    total_tags = sum(tag_counts.values())
    if total_tags == 0:
        return {'pos_freq': {}}

    pos_freq = {tag: count / total_tags for tag, count in tag_counts.items()}

    if top_n:
        pos_freq = dict(sorted(pos_freq.items(), key=lambda item: item[1], reverse=True)[:top_n])
    return {'pos_freq': pos_freq}

def calculate_punct_freq_spacy(spacy_tokens):
    """Calculates punctuation frequencies (per 1000 tokens) using spaCy token.is_punct."""
    if not spacy_tokens:
        return {'punct_freq': {}}
    # Use spaCy's is_punct attribute
    punct_counts = Counter(token.text for token in spacy_tokens if token.is_punct)
    total_tokens = len(spacy_tokens) # Count all tokens (words+punct) for normalization base
    if total_tokens == 0:
        return {'punct_freq': {}}
    # Normalize to frequency per 1000 tokens
    punct_freq = {punct: (count / total_tokens) * 1000 for punct, count in punct_counts.items()}
    return {'punct_freq': punct_freq}

def calculate_readability(full_text_string):
    """Calculates Flesch Reading Ease score."""
    if not full_text_string or not isinstance(full_text_string, str):
        return {'flesch_ease': None}
    try:
        ease = textstat.flesch_reading_ease(full_text_string)
        return {'flesch_ease': ease}
    except Exception as e:
        print(f"Warning: Could not calculate readability: {e}")
        return {'flesch_ease': None}

# --- Main Extractor Function ---

def extract_all_features(text: str) -> dict:
    """
    Preprocesses raw text using spaCy and extracts a dictionary of stylometric features.
    """
    global NLP # Use the globally loaded spaCy model
    if NLP is None:
        print("ERROR: spaCy model was not loaded successfully. Cannot extract features.")
        return {}
    if not isinstance(text, str) or not text.strip():
        print("Warning: Input text is empty or not a string.")
        return {}

    # 1. Preprocessing with spaCy (and NLTK sentence splitting for consistency)
    # It's generally more efficient to process larger texts with spaCy
    # but splitting first might handle very large inputs better.
    # Alternatively, process full text: doc = NLP(text) then use doc.sents
    try:
        # Use NLTK for sentence splitting first
        initial_sentences = sent_tokenize(text)
        if not initial_sentences:
            print("Warning: No sentences found after initial splitting.")
            return {}

        all_spacy_tokens = []
        processed_sentences = []

        # Process sentence by sentence with spaCy
        # Note: processing the whole text `doc = NLP(text)` then iterating `doc.sents`
        # might be more efficient for spaCy's internal processing if text isn't huge.
        for sent_text in initial_sentences:
             sent_text_clean = re.sub(r'\s+', ' ', sent_text).strip()
             if not sent_text_clean: continue
             # Process sentence with spaCy
             doc = NLP(sent_text_clean)
             sentence_tokens = [token for token in doc if not token.is_space]
             all_spacy_tokens.extend(sentence_tokens)
             processed_sentences.append(sent_text_clean) # Store original sentence text for stats

        # Need the full text for readability score - reconstruct or use original
        # Using original text might be more accurate for textstat
        full_processed_text = text # Use original text passed to the function

    except LookupError as e:
         print(f"ERROR: NLTK 'punkt' data not found for sentence tokenization: {e}. Cannot process.")
         print("Run: python -m nltk.downloader punkt")
         return {}
    except Exception as e:
         print(f"Error during preprocessing or spaCy processing: {e}")
         return {}

    if not all_spacy_tokens:
        print("Warning: No tokens found after spaCy processing.")
        return {}

    # 2. Feature Calculation using spaCy tokens where appropriate
    print(f"Extracting features from {len(all_spacy_tokens)} spaCy tokens...")
    features = {}
    # Sentence stats based on initial NLTK splitting
    features.update(calculate_sentence_stats(processed_sentences))
    # Other stats based on accumulated spaCy tokens
    features.update(calculate_word_stats(all_spacy_tokens))
    features.update(calculate_ttr(all_spacy_tokens))
    features.update(calculate_pos_freq_spacy(all_spacy_tokens)) # Use spaCy POS func
    features.update(calculate_punct_freq_spacy(all_spacy_tokens)) # Use spaCy Punct func
    features.update(calculate_readability(full_processed_text)) # Use original text

    # Filter out None values
    features = {k: v for k, v in features.items() if v is not None}
    print("Feature extraction complete.")
    return features

# --- Example Usage (if script is run directly) ---
if __name__ == '__main__':
    sample_text = """This is the first sentence. This, the second sentence, is a bit longer. And a third! How about that?"""
    print(f"Sample Text:\n{sample_text}\n")

    # Only proceed if spaCy model loaded
    if NLP:
        extracted_features = extract_all_features(sample_text)
        print("\nExtracted Features:")
        # Use json for pretty printing the dictionary
        print(json.dumps(extracted_features, indent=2))
    else:
        print("\nCannot run example usage because spaCy model failed to load.")