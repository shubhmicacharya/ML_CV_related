import pdfplumber
import re
import json
import pandas as pd
import spacy
from nltk.corpus import stopwords
import nltk
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text


def extract_section(text, section_keywords):
    extracted = {}
    text = clean_text(text)

    for section, keywords in section_keywords.items():
        extracted[section] = []
        for keyword in keywords:
            pattern = rf"{keyword}\s*:?\s*(.*?)(?=\n\s*\n|$)"
            matches = re.findall(pattern, text, re.S)
            if matches:
                extracted[section].extend(matches)

    return extracted


def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [str(word) for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def save_to_json(data, filename):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)


def save_to_csv(data, file_name):
    df = pd.DataFrame.from_dict(data, orient="index").transpose()
    df.to_csv(file_name, index=False)


section_keywords = {
    "Languages": ["languages:"],
    "Libraries": ["libraries:"],
    "Databases": ["databases:"],
    "Data Analytics Tools": ["data analytics tools:"],
    "Other Tools": ["other tools:"],
    "Relevant Coursework": ["relevant coursework:"],
    "Areas of Interest": ["areas of interest:"],
    "Soft Skills": ["soft skills:"]
}

pdf_path = "Shubh_RESUME (4).pdf"
pdf_text = extract_text(pdf_path)
cleaned_text = clean_text(pdf_text)
extracted_data = extract_section(cleaned_text, section_keywords)

print("Extracted Data:", extracted_data)

save_to_json(extracted_data, "resume_data.json")
save_to_csv(extracted_data, "resume_data.csv")
print("Data successfully saved in JSON and CSV formats")
