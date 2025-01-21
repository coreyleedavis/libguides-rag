import re

# Input and output file paths
input_file = 'scraped.txt'  # Replace with the correct input file path
output_file = 'cleaned.txt'

def clean_text(text):
    """Cleans the text by removing unwanted characters while preserving whitespace and structure."""
    # Remove special characters except for alphanumeric, basic punctuation, and whitespace
    text = re.sub(r'[^\w\s.,;!?\'"-]', '', text)  # Keeps alphanumeric, punctuation, and whitespace
    # Normalize multiple spaces to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    # Retain paragraph breaks (normalize multiple newlines)
    text = re.sub(r'(\n\s*)+', '\n', text)
    return text

def process_file(input_path, output_path):
    """Processes the input text file and writes cleaned text to the output file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            raw_text = infile.read()

        # Clean and refine text
        cleaned_text = clean_text(raw_text)

        # Write the cleaned text to the output file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned_text)

        print(f"Cleaned text saved to: {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    process_file(input_file, output_file)
