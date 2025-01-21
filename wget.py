import os
import subprocess

def download_as_warc(url, warc_file_name, exclude_extensions):
    """
    Downloads a website and saves it as a WARC file, excluding specific file types.

    :param url: The URL of the website to download.
    :param warc_file_name: The name of the WARC file to create (without extension).
    :param exclude_extensions: A list of file extensions to exclude from the download.
    """
    # Join exclude extensions into a comma-separated string for wget
    exclude_str = ",".join([f"*.{ext}" for ext in exclude_extensions])

    # Build the wget command
    command = [
        "wget",
        "-r",  # Recursive download
        "-R", exclude_str,  # Reject specified file types
        "--warc-file", warc_file_name,  # Output WARC file
        "--warc-cdx",  # Generate CDX index
        "-np",  # No parent directories
        "--delete-after",  # Delete downloaded files after saving to WARC
        url  # Target website
    ]

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Website archived successfully as {warc_file_name}.warc.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading: {e}")

if __name__ == "__main__":
    # Example usage:
    # Replace the following with your desired URL and output file name
    website_url = "https://libguides.uvic.ca/"
    warc_name = "libguide_archive"

    # List of file extensions to exclude
    excluded_types = [
        "css", "js", "jpg", "jpeg", "png", "gif", "bmp", "svg", "ico", "webp", "webm", 
        "mp4", "mp3", "wav", "avi", "mov", "wmv", "flv", "mkv", "ogg", "m4a", "m4v", 
        "flac", "midi", "mpg", "mpeg", "json", "xml", "rss", "woff", "woff2", "ttf", 
        "otf", "eot"
    ]

    # Call the function
    download_as_warc(website_url, warc_name, excluded_types)