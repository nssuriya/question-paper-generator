
from PyPDF2 import PdfReader

def read_pdf():
    with open("Class10.pdf", "rb") as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def main():
    text = read_pdf()
    print("Extracted Text:\n")
    print(text)
if __name__ == "__main__":
    main()
# This code reads a PDF file and extracts its text content using PyPDF2.
