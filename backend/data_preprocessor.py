# backend/data_preprocessor.py
import xml.etree.ElementTree as ET
import json
import nltk
import os
from bs4 import BeautifulSoup  # Import BeautifulSoup

# --- Configuration ---
RAW_XML_PATH = os.path.join(
    os.path.dirname(__file__), "data", "medlineplus", "mplus_topics_2025-05-27.xml"
)
PROCESSED_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "data", "medlineplus", "processed_health_topics.json"
)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# MODIFIED FUNCTION using BeautifulSoup
def extract_text_with_bs(element_or_string_content):
    if element_or_string_content is None:
        return ""

    # If it's an ElementTree element, convert its inner HTML content to a string
    if hasattr(element_or_string_content, "tag"):  # Check if it's an ET element
        html_parts = []
        if element_or_string_content.text:  # Process text before first child
            html_parts.append(element_or_string_content.text)
        for child in element_or_string_content:  # Process children and their tails
            html_parts.append(ET.tostring(child, encoding="unicode", method="html"))
            # Note: ET.tostring for a child includes the child's tail already if method='html'
        html_content = "".join(html_parts)
    else:  # If it's already a string (e.g., from element.text directly)
        html_content = str(element_or_string_content)  # Ensure it's a string

    if not html_content.strip():
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    # Get text, using a space as a separator between tags, and strip leading/trailing whitespace
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())  # Normalize whitespace again


def chunk_text_by_sentences(text, sentences_per_chunk=5, overlap_sentences=2):
    sentences = nltk.sent_tokenize(text)
    if not sentences: return []
    chunks = []
    step = sentences_per_chunk - overlap_sentences
    if step <= 0: step = 1 # Ensure progress
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)
        if i + sentences_per_chunk >= len(sentences):
            break
    return chunks


def process_medlineplus_xml(xml_file_path, output_json_path):
    print(f"Processing MedlinePlus XML: {xml_file_path}")
    health_topics_data = []

    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(
            f"ERROR: XML file not found at {xml_file_path}. Please download it first."
        )
        return
    except ET.ParseError as e:
        print(f"ERROR: Could not parse XML file: {e}")
        return

    for topic_element in root.findall("health-topic"):
        title = topic_element.get("title", "No Title")
        url = topic_element.get("url", "No URL")

        content_text = ""
        full_summary_element = topic_element.find("full-summary")
        if full_summary_element is not None:
            # USE THE NEW BS FUNCTION HERE
            content_text = extract_text_with_bs(full_summary_element)

        if not content_text:  # Fallback if full-summary was empty or not found
            summary_element = topic_element.find("summary")
            if summary_element is not None:
                # USE THE NEW BS FUNCTION HERE
                content_text = extract_text_with_bs(summary_element)

        if not content_text:
            print(
                f"Skipping topic '{title}' due to no substantial text content found in <full-summary> or <summary>."
            )
            continue

        # BeautifulSoup's get_text() also handles HTML unescaping implicitly.
        # So, `html.unescape()` is generally not needed here.

        text_chunks = chunk_text_by_sentences(content_text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(text_chunks):
            topic_id_attr = topic_element.get(
                "id", url
            )  # Use topic 'id' attribute if present
            chunk_id = f"{topic_id_attr}#{i}"
            health_topics_data.append(
                {
                    "id": chunk_id,
                    "title": title,
                    "url": url,
                    "content_chunk": chunk,
                    "source": "MedlinePlus",
                }
            )

        if text_chunks:
            print(f"Processed topic: '{title}', extracted {len(text_chunks)} chunk(s).")

    if health_topics_data:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(health_topics_data, f, indent=4, ensure_ascii=False)
        print(
            f"Successfully processed {len(root.findall('health-topic'))} topics, yielding {len(health_topics_data)} text chunks."
        )
        print(f"Processed data saved to: {output_json_path}")
    else:
        print(
            "No data was processed. Please check the XML structure and extraction logic."
        )


if __name__ == "__main__":
    os.makedirs(
        os.path.dirname(RAW_XML_PATH), exist_ok=True
    )  # Ensure data/medlineplus directory exists
    if not os.path.exists(RAW_XML_PATH):
        print(f"MedlinePlus XML file not found at: {RAW_XML_PATH}")
        print(
            "Please download 'mplus_topics_YYYY-MM-DD.xml' from https://medlineplus.gov/xml.html,"
        )
        print(f"ensure its name matches '{os.path.basename(RAW_XML_PATH)}',")
        print("and place it in the 'backend/data/medlineplus/' directory.")
    else:
        process_medlineplus_xml(RAW_XML_PATH, PROCESSED_JSON_PATH)
