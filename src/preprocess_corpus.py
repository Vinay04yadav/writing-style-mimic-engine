import os
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import time

# --- Configuration ---
DATA_RAW_DIR = "/Users/vinayyadav/Writing_Style_Mimic/data/raw"
DATA_PROCESSED_DIR = "/Users/vinayyadav/Writing_Style_Mimic/data/processed"

# --- Preprocessing Steps Configuration ---
DO_LOWERCASE = True
# Add other flags here if you decide to add more steps later (e.g., DO_LEMMATIZE = False)

print("Preprocessing script started.")
print(f"Reading from: {os.path.abspath(DATA_RAW_DIR)}")
print(f"Writing to: {os.path.abspath(DATA_PROCESSED_DIR)}")

# Ensure the main processed directory exists
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Check if NLTK 'punkt' data is available, provide instructions if not
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer found.")
except LookupError:
    print("ERROR: NLTK 'punkt' tokenizer data not found.")
    print("Please download it by running this command in your terminal:")
    print("python -m nltk.downloader punkt")
    exit() # Stop the script if essential data is missing

start_time = time.time()

# Get list of author directories
try:
    author_folders = [d for d in os.listdir(DATA_RAW_DIR) if os.path.isdir(os.path.join(DATA_RAW_DIR, d))]
    if not author_folders:
        print(f"ERROR: No author folders found in {DATA_RAW_DIR}. Did the download script run correctly?")
        exit()
except FileNotFoundError:
    print(f"ERROR: Raw data directory not found: {DATA_RAW_DIR}")
    print("Make sure you have run the download script first and the path is correct.")
    exit()


print(f"Found author folders: {author_folders}")

# Process each author
for author_folder_name in author_folders:
    print(f"\nProcessing author: {author_folder_name}...")
    raw_author_dir = os.path.join(DATA_RAW_DIR, author_folder_name)
    processed_author_dir = os.path.join(DATA_PROCESSED_DIR, author_folder_name)

    # Create the processed directory for the author
    os.makedirs(processed_author_dir, exist_ok=True)

    all_processed_sentences_for_author = []
    file_count = 0

    # Iterate through files in the raw author directory
    try:
        for filename in os.listdir(raw_author_dir):
            # Process only .txt files
            if filename.lower().endswith(".txt"):
                file_path = os.path.join(raw_author_dir, filename)
                print(f"--- Reading file: {filename}")
                file_count += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_text = f.read()

                    # 1. Sentence Tokenization
                    sentences = sent_tokenize(raw_text)

                    # 2. Process each sentence
                    for sentence in sentences:
                        # Remove extra whitespace within sentence before tokenizing
                        sentence = re.sub(r'\s+', ' ', sentence).strip()
                        if not sentence: # Skip empty sentences
                            continue

                        # 3. Word Tokenization
                        word_tokens = word_tokenize(sentence)

                        # 4. Lowercasing (optional)
                        if DO_LOWERCASE:
                            processed_tokens = [token.lower() for token in word_tokens]
                        else:
                            processed_tokens = word_tokens

                        # 5. Join tokens back into a processed sentence string
                        processed_sentence = ' '.join(processed_tokens)
                        all_processed_sentences_for_author.append(processed_sentence)

                except FileNotFoundError:
                    print(f"--- Warning: File not found during processing (should not happen): {file_path}")
                except Exception as e:
                    print(f"--- Error processing file {filename}: {e}")
                    # Optionally skip the file or handle specific errors

    except FileNotFoundError:
         print(f"--- Error: Directory not found {raw_author_dir}. Skipping author.")
         continue # Skip to the next author

    # Write the processed corpus for the author
    if all_processed_sentences_for_author:
        output_corpus_path = os.path.join(processed_author_dir, "corpus.txt")
        print(f"--- Writing processed corpus for {author_folder_name} ({len(all_processed_sentences_for_author)} sentences from {file_count} files) to {output_corpus_path}")
        try:
            with open(output_corpus_path, 'w', encoding='utf-8') as f_out:
                # Write one processed sentence per line
                f_out.write('\n'.join(all_processed_sentences_for_author))
            print(f"--- Successfully wrote corpus for {author_folder_name}.")
        except Exception as e:
             print(f"--- Error writing processed corpus file {output_corpus_path}: {e}")
    else:
        print(f"--- Warning: No text processed for author {author_folder_name}. No corpus file written.")


end_time = time.time()
print(f"\nPreprocessing finished in {end_time - start_time:.2f} seconds.")