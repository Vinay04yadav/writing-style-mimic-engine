# src/authorship_analyzer.py

import os
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler # For normalizing features
import sys
import argparse

# Ensure src directory is in path if running script directly (or use relative imports)
# This allows importing from stylometry_features when running as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from stylometry_features import extract_all_features
except ImportError:
    print("ERROR: Could not import 'extract_all_features' from 'stylometry_features.py'.")
    print("Ensure both scripts are in the 'src' directory.")
    sys.exit(1)

# --- Configuration ---
# Path to the saved author profiles (relative to project root assumed)
PROFILES_PATH = os.path.join("data", "analysis_results", "author_profiles.pkl")

# Define the features to use for comparison vector
# Numerical features first
NUMERICAL_FEATURES = [
    'avg_sent_len', 'std_sent_len', 'avg_word_len', 'ttr', 'flesch_ease'
]
# Top N POS tags (find these from your analysis notebook or define a standard set)
TOP_N_POS = 15 # Example
POS_FEATURE_KEYS = [] # We'll populate this dynamically later
# Top N Punctuation marks
TOP_N_PUNCT = 10 # Example
PUNCT_FEATURE_KEYS = [] # We'll populate this dynamically later

# --- Helper Functions ---

def load_author_profiles(path=PROFILES_PATH):
    """Loads the pre-calculated author profiles DataFrame."""
    if not os.path.exists(path):
        print(f"ERROR: Author profiles file not found at {path}")
        return None
    try:
        with open(path, 'rb') as f:
            df_profiles = pickle.load(f)
        print(f"Loaded author profiles from {path}")
        return df_profiles
    except Exception as e:
        print(f"Error loading author profiles: {e}")
        return None

def determine_feature_keys(df_profiles):
    """Dynamically determine top POS and Punctuation keys from profiles."""
    global POS_FEATURE_KEYS, PUNCT_FEATURE_KEYS

    # POS Keys
    all_pos_tags = Counter()
    for profile in df_profiles['pos_freq']:
        all_pos_tags.update(profile.keys())
    POS_FEATURE_KEYS = [tag for tag, count in all_pos_tags.most_common(TOP_N_POS)]
    print(f"Using Top {len(POS_FEATURE_KEYS)} POS tags: {POS_FEATURE_KEYS}")

    # Punctuation Keys
    all_punct = Counter()
    for profile in df_profiles['punct_freq']:
         # Filter out potentially weird keys if necessary before counting
         valid_puncts = {k:v for k,v in profile.items() if isinstance(k, str) and len(k) < 5}
         all_punct.update(valid_puncts.keys())
    PUNCT_FEATURE_KEYS = [p for p, count in all_punct.most_common(TOP_N_PUNCT)]
    print(f"Using Top {len(PUNCT_FEATURE_KEYS)} Punctuation marks: {PUNCT_FEATURE_KEYS}")


def profile_to_vector(profile: dict, numerical_features, pos_keys, punct_keys):
    """Converts a feature profile dictionary into a fixed-length numerical vector."""
    vector = []
    # Add numerical features
    for feature in numerical_features:
        vector.append(profile.get(feature, 0)) # Use 0 if feature missing

    # Add POS features
    pos_profile = profile.get('pos_freq', {})
    for key in pos_keys:
        vector.append(pos_profile.get(key, 0)) # Use 0 frequency if tag missing

    # Add Punctuation features
    punct_profile = profile.get('punct_freq', {})
    for key in punct_keys:
        vector.append(punct_profile.get(key, 0)) # Use 0 frequency if punct missing

    return np.array(vector)

# --- Main Analysis Function ---

def analyze_authorship(unknown_text: str, df_profiles: pd.DataFrame, scaler: MinMaxScaler):
    """
    Analyzes unknown text and compares its style vector to known author profiles.
    Returns the most likely author and similarity scores.
    """
    if not isinstance(unknown_text, str) or not unknown_text:
        return "Error: Input text is invalid.", {}

    # 1. Extract features from unknown text
    unknown_features = extract_all_features(unknown_text)
    if not unknown_features:
        return "Error: Could not extract features from input text.", {}

    # 2. Convert unknown features to vector
    unknown_vector = profile_to_vector(
        unknown_features, NUMERICAL_FEATURES, POS_FEATURE_KEYS, PUNCT_FEATURE_KEYS
    )

    # 3. Scale the unknown vector using the *fitted* scaler
    # Reshape because scaler expects 2D array
    unknown_vector_scaled = scaler.transform(unknown_vector.reshape(1, -1))[0]

    # 4. Calculate similarity to each known author
    similarities = {}
    for author_name, author_profile_series in df_profiles.iterrows():
        author_profile_dict = author_profile_series.to_dict()
        # Get the pre-scaled vector for the author
        author_vector_scaled = author_profile_dict.get('scaled_vector')
        if author_vector_scaled is None or not isinstance(author_vector_scaled, np.ndarray):
             print(f"Warning: Scaled vector missing or invalid for author {author_name}. Skipping.")
             continue

        # Calculate cosine similarity (1 - cosine distance)
        # Ensure vectors are not zero vectors
        if np.any(unknown_vector_scaled) and np.any(author_vector_scaled):
            similarity = 1 - cosine(unknown_vector_scaled, author_vector_scaled)
        else:
            similarity = 0 # Or handle as error/undefined?
        similarities[author_name] = similarity

    # 5. Determine most likely author
    if not similarities:
        return "Error: Could not compare with any author profiles.", {}

    most_likely_author = max(similarities, key=similarities.get)

    return most_likely_author, similarities


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze authorship of an unknown text snippet.")
    parser.add_argument("text_snippet", type=str, help="The text snippet to analyze.")
    # parser.add_argument("-f", "--file", type=str, help="Path to a text file containing the snippet.") # Optional file input

    args = parser.parse_args()
    unknown_text_input = args.text_snippet

    # --- Initialization ---
    df_author_profiles = load_author_profiles()

    if df_author_profiles is not None:
        # Determine feature keys dynamically based on loaded profiles
        determine_feature_keys(df_author_profiles)
        FEATURE_SET = NUMERICAL_FEATURES + POS_FEATURE_KEYS + PUNCT_FEATURE_KEYS

        # Create and fit the scaler ONCE on the known author profiles
        known_author_vectors = np.array([
            profile_to_vector(prof.to_dict(), NUMERICAL_FEATURES, POS_FEATURE_KEYS, PUNCT_FEATURE_KEYS)
            for idx, prof in df_author_profiles.iterrows()
        ])

        # Check if we have valid vectors before scaling
        if known_author_vectors.size > 0:
            scaler = MinMaxScaler()
            scaled_known_vectors = scaler.fit_transform(known_author_vectors)

            # Store scaled vectors back into the DataFrame (or a separate dict) for easy access
            df_author_profiles['scaled_vector'] = list(scaled_known_vectors)

            # --- Run Analysis ---
            predicted_author, all_scores = analyze_authorship(unknown_text_input, df_author_profiles, scaler)

            print("\n--- Authorship Analysis Result ---")
            print(f"Input Text:\n'{unknown_text_input[:200]}...'") # Print snippet
            print(f"\nPredicted Author: {predicted_author}")
            print("\nSimilarity Scores:")
            # Sort scores highest first
            sorted_scores = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
            for author, score in sorted_scores:
                print(f"- {author}: {score:.4f}")
        else:
             print("Error: Could not create valid feature vectors from author profiles.")

    else:
        print("Could not proceed without author profiles.")