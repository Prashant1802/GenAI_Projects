
import fitz # fitz is the Python module provided by PyMuPDF. It helps us work with PDF files.
from pathlib import Path

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define input and output paths
PDF_PATH = PROJECT_ROOT / "data" / "documents" / "sample_policy.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_text"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):

    extracted_text = ""

    # Open the PDF
    with fitz.open(pdf_path) as pdf:

        print(f"Total pages: {len(pdf)}")

        # Read each page
        for page_number, page in enumerate(pdf, start=1):

            # It extracts the text available on that particular PDF page
            page_text = page.get_text("text")

            print(f"Extracted page {page_number}")

            extracted_text += f"\n\n--- Page {page_number} ---\n\n"
            extracted_text += page_text

    return extracted_text


if __name__ == "__main__":

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Extract text
    text = extract_text_from_pdf(PDF_PATH)

    # Save extracted text
    output_file = OUTPUT_DIR / "extracted_text.txt"

    # This saves the extracted content into a text file
    output_file.write_text(text, encoding="utf-8")

    print("\nText extraction completed!")
    print(f"Extracted characters: {len(text)}")
    print(f"Output saved at: {output_file}")

    # Display first 1000 characters
    print("\nPreview:")
    print(text[:1000])
