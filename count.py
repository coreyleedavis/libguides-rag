def count_words_in_file(file_path):
    """Counts the number of words in a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            # Split the text into words
            words = text.split()
            return len(words)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def estimate_tokens(word_count, factor=1.2):
    """
    Estimates the number of tokens based on the word count.
    A rough factor is applied, typically ~1.2 tokens per word.
    """
    return int(word_count * factor)

if __name__ == "__main__":
    # Path to the input file
    file_path = "cleaned.txt"  # Provide the file name directly
    word_count = count_words_in_file(file_path)
    if word_count is not None:
        print(f"The number of words in the file is: {word_count}")
        
        # Estimate the token count
        token_count = estimate_tokens(word_count)
        print(f"Estimated number of tokens: {token_count}")
