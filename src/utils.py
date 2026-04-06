import pdfplumber
import os
import requests
from bs4 import BeautifulSoup
from test import scrape_job_posting

 
def get_pdf_text(pdf_path, layout=False):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: '{pdf_path}'")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=layout)
                print(text)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}") from e



def get_job_description_basic(link):
    response = requests.get("https://example.com")
    soup = BeautifulSoup(response.text, "html.parser")
if __name__ == "__main__":
    resume_path = 'resume_zasha_benites.pdf'


