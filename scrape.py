from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
from langdetect import detect
import os
import gzip

# Path to the WARC file
input_warc_file = 'libguide_archive.warc.gz'  # Replace with your WARC file path
output_text_file = 'scraped.txt'

def is_informative(text):
    """Checks if the paragraph is likely to contain meaningful information."""
    noise_keywords = ['click', 'advertisement', 'subscribe', 'privacy', 'cookie']
    return (
        len(text.split()) > 5 and  # Ensure minimum word count
        not any(keyword in text.lower() for keyword in noise_keywords)  # Filter out noise
    )

def is_desired_language(text, lang='en'):
    """Checks if the text is in the desired language."""
    try:
        return detect(text) == lang
    except:
        return False

def extract_text_with_headings(html_content):
    """Extracts text grouped by headings for better contextual chunking."""
    soup = BeautifulSoup(html_content, 'html.parser')
    output = []
    current_heading = None
    buffer = []

    for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
        if tag.name.startswith('h'):
            if buffer:
                output.append({'heading': current_heading, 'content': ' '.join(buffer)})
                buffer = []
            current_heading = tag.get_text()
        elif tag.name == 'p' and is_informative(tag.get_text()):
            buffer.append(tag.get_text())

    if buffer:  # Add any remaining paragraphs
        output.append({'heading': current_heading, 'content': ' '.join(buffer)})

    return output

def chunk_text(text, max_tokens=500):
    """Chunks text into manageable sizes while keeping semantic boundaries."""
    sentences = text.split('. ')
    chunks, current_chunk = [], []
    token_count = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        if token_count + tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            token_count = 0
        current_chunk.append(sentence)
        token_count += tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_warc_file(warc_file, output_file):
    """Processes a WARC file to extract text from main content areas and save it to a file."""
    with open(output_file, 'w', encoding='utf-8') as output:
        with gzip.open(warc_file, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type == 'response':
                    try:
                        payload = record.content_stream().read()
                        html_content = payload.decode('utf-8', errors='ignore')

                        # Extract and process text
                        sections = extract_text_with_headings(html_content)
                        for section in sections:
                            if section['content'] and is_desired_language(section['content']):
                                chunks = chunk_text(section['content'])
                                for chunk in chunks:
                                    output.write(f"Heading: {section['heading']}\n")
                                    output.write(f"Content:\n{chunk}\n")
                                    output.write("="*80 + "\n")

                    except Exception as e:
                        print(f"Error processing record: {e}")

if __name__ == '__main__':
    if not os.path.exists(input_warc_file):
        print(f"WARC file '{input_warc_file}' not found.")
    else:
        print(f"Processing WARC file: {input_warc_file}")
        process_warc_file(input_warc_file, output_text_file)
        print(f"Text extracted and saved to: {output_text_file}")
