import os
import requests
import time
import re # Added for manual cleaning function

# --- Helper Function for Manual Cleaning ---

def clean_gutenberg_text_manually(text):
    """Removes Project Gutenberg headers/footers using common markers."""
    # More robust patterns using regex, ignoring case
    start_marker_patterns = [
        r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*",
    ]
    end_marker_patterns = [
        r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*",
        r"End of the Project Gutenberg EBook of .*",
        r"End of Project Gutenberg's .*",
    ]

    start_pos = -1
    # Find the latest occurrence of any start marker
    for pattern in start_marker_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            start_pos = max(start_pos, matches[-1].end())

    end_pos = len(text)
    found_end_marker = False
    # Find the earliest occurrence of any end marker
    for pattern in end_marker_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            end_pos = min(end_pos, matches[0].start())
            found_end_marker = True

    # Extract text after the latest start marker found
    if start_pos != -1:
        # Adjust start position (often need to skip newline after marker)
        current_char = text[min(start_pos, len(text)-1)]
        if current_char == '\n' or current_char == '\r':
             start_pos +=1
        elif start_pos + 1 < len(text): # Check next char if not newline
            next_char = text[start_pos+1]
            if next_char == '\n' or next_char == '\r':
                 start_pos +=2 # Skip marker and newline sequence

        # Ensure start_pos doesn't exceed text length
        start_pos = min(start_pos, len(text))

        text_after_start = text[start_pos:]
        # Adjust end_pos relative to the start position
        relative_end_pos = end_pos - start_pos
    else:
        print("--- Warning: Gutenberg start marker not found. Cleaning might be incomplete.")
        text_after_start = text # Use original text if no start marker
        relative_end_pos = end_pos # Use absolute end position

    # Extract text before the earliest end marker found
    if found_end_marker and relative_end_pos >= 0: # Check relative_end_pos >= 0
         cleaned_text = text_after_start[:relative_end_pos]
    else:
         if not found_end_marker:
             print("--- Warning: Gutenberg end marker not found. Cleaning might be incomplete.")
         cleaned_text = text_after_start # Use text after start marker if no end marker

    # Basic whitespace stripping
    return cleaned_text.strip()

# --- Configuration ---

# Define the base directory where raw data will be stored
# Assumes you run this script from the root of your project folder (e.g., Writing_Style_Mimic)
DATA_RAW_DIR = "data/raw"

# List of books to download:
# (Using your provided list)
BOOKS_TO_DOWNLOAD = [
    {
        'author': 'William Shakespeare',
        'book_id': 16966,
        'title_for_filename': 'Shakespearean Tragedy: Lectures on Hamlet, Othello, King Lear, Macbeth by Bradley'
    },
    {
        'author': 'William Shakespeare',
        'book_id': 23042,
        'title_for_filename': 'The Tempest'
    },
    {
        'author': 'William Shakespeare',
        'book_id': 1514,
        'title_for_filename': 'A Midsummer Nights Dream'
    },
    {
        'author': 'William Shakespeare',
        'book_id': 1524,
        'title_for_filename': 'Hamlet, Prince of Denmark'
    },
    {
        'author': 'William Shakespeare',
        'book_id': 1531,
        'title_for_filename': 'Othello, the Moor of Venice'
    },

    {
        'author': 'Jane Austen',
        'book_id': 105,
        'title_for_filename': 'Persuasion'
    },
    {
        'author': 'Jane Austen',
        'book_id': 121,
        'title_for_filename': 'Northanger Abbey'
    },
    {
        'author': 'Jane Austen',
        'book_id': 141,
        'title_for_filename': 'Mansfield Park'
    },
    {
        'author': 'Jane Austen',
        'book_id': 42671,
        'title_for_filename': 'Pride and Prejudice'
    },
    {
        'author': 'Jane Austen',
        'book_id': 19839,
        'title_for_filename': 'Emma'
    },

    {
        'author': 'Charles Dickens',
        'book_id': 786,
        'title_for_filename': 'Hard Times'
    },
    {
        'author': 'Charles Dickens',
        'book_id': 24022,
        'title_for_filename': 'A Christmas Carol'
    },
    {
        'author': 'Charles Dickens',
        'book_id': 1023,
        'title_for_filename': 'Bleak House'
    },
    {
        'author': 'Charles Dickens',
        'book_id': 730,
        'title_for_filename': 'Oliver Twist'
    },
    {
        'author': 'Charles Dickens',
        'book_id': 766,
        'title_for_filename': 'David Copperfield'
    },

    {
        'author': 'Arthur Conan Doyle',
        'book_id': 108,
        'title_for_filename': 'The Return of Sherlock Holmes'
    },
    {
        'author': 'Arthur Conan Doyle',
        'book_id': 48320,
        'title_for_filename': 'Adventures of Sherlock Holmes'
    },
    {
        'author': 'Arthur Conan Doyle',
        'book_id': 3289,
        'title_for_filename': 'The Valley of Fear'
    },
    {
        'author': 'Arthur Conan Doyle',
        'book_id': 2097,
        'title_for_filename': 'The Sign of the Four'
    },
    {
        'author': 'Arthur Conan Doyle',
        'book_id': 834,
        'title_for_filename': 'The Memoirs of Sherlock Holmes'
    },

    {
        'author': 'Agatha Christie',
        'book_id': 61262,
        'title_for_filename': 'Poirot Investigates'
    },
    {
        'author': 'Agatha Christie',
        'book_id': 58866,
        'title_for_filename': 'The Murder on the Links'
    },
    {
        'author': 'Agatha Christie',
        'book_id': 1155,
        'title_for_filename': 'The Secret Adversary'
    },
    {
        'author': 'Agatha Christie',
        'book_id': 75288,
        'title_for_filename': 'The Seven Dials mystery'
    },
    {
        'author': 'Agatha Christie',
        'book_id': 61168,
        'title_for_filename': 'The Man in the Brown Suit'
    },
]

# Delay between downloads (in seconds) to be polite to Project Gutenberg's servers
DOWNLOAD_DELAY = 2

# --- Script Logic ---

print("Starting corpus download process...")

for book_info in BOOKS_TO_DOWNLOAD:
    author = book_info['author']
    book_id = book_info['book_id']
    # *** Assign title_slug from the dictionary ***
    title_slug = book_info['title_for_filename']

    print(f"\nProcessing: {author} - {title_slug} (ID: {book_id})")

    # Clean up title for a safe filename
    safe_title = re.sub(r'[\\/*?:"<>|\']', "", title_slug) # Remove illegal filename chars (added apostrophe)
    safe_title = safe_title.replace(' ', '_').lower()     # Replace spaces, lowercase
    safe_title = safe_title[:50] # Limit length to avoid issues

    # 1. Create author directory if it doesn't exist
    author_folder_name = author.lower().replace(' ', '_')
    author_dir = os.path.join(DATA_RAW_DIR, author_folder_name)
    os.makedirs(author_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

    # 2. Define output path and check if file already exists
    output_filename = f"{book_id}_{safe_title}.txt"
    output_path = os.path.join(author_dir, output_filename)

    if os.path.exists(output_path):
        print(f"--- Skipping: File already exists at {output_path}")
        continue # Move to the next book in the list

    # 3. Construct potential download URLs (try -0.txt first, then -8.txt)
    url_patterns = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt", # Common UTF-8
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-8.txt"  # Another common UTF-8 variant
    ]

    raw_text = None
    download_url = None

    # 4. Attempt download
    for url in url_patterns:
        try:
            print(f"--- Attempting download from: {url}")
            response = requests.get(url, timeout=60) # Increased timeout to 60s
            response.raise_for_status() # Raise HTTPError for bad responses (4XX, 5XX)

            content_type = response.headers.get('content-type', '').lower()
            if 'text/plain' in content_type:
                print("--- Download successful (Status Code 200, Content-Type text/plain)")
                # Decode explicitly using UTF-8, ignore errors for robustness
                raw_text = response.content.decode('utf-8', errors='ignore')
                download_url = url
                break
            else:
                 print(f"--- Warning: Status Code 200 but Content-Type is '{content_type}'. Trying next URL.")

        except requests.exceptions.HTTPError as http_err:
            print(f"--- HTTP error occurred: {http_err} (Status code: {response.status_code})")
            if response.status_code == 404:
                print("--- File not found at this URL, trying next pattern...")
                continue
            else:
                print("--- Unrecoverable HTTP error, stopping download attempt for this book.")
                break
        except requests.exceptions.Timeout:
             print("--- Error: The request timed out.")
             break
        except requests.exceptions.RequestException as req_err:
            print(f"--- Network error occurred: {req_err}")
            break
        finally:
             # Add delay *before* starting next attempt or moving to next book
             print(f"--- Waiting for {DOWNLOAD_DELAY} seconds...")
             time.sleep(DOWNLOAD_DELAY)


    # 5. Clean and save if download was successful
    if raw_text:
        print("--- Cleaning text (removing headers/footers)...")
        try:
            # *** Use the manual cleaning function ***
            clean_text = clean_gutenberg_text_manually(raw_text)

            if not clean_text:
                print("--- Warning: Cleaning resulted in empty text. Check raw download.")

            print(f"--- Saving cleaned text to: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print("--- File saved successfully.")

        except Exception as e:
             import traceback
             print(f"--- Error during cleaning or saving file {output_path}: {e}")
             traceback.print_exc()

    else:
        print(f"--- Failed to download book {book_id} ('{title_slug}') after trying all patterns.")

print("\nCorpus download process finished.")