import fitz

def extract_text_from_pdf(pdf_name):
    doc = fitz.open(pdf_name)
    text = ""

    for page in doc:
        text += page.get_text()
    return text

def resume_classifier(text):
    from transformers import pipeline
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    result = classifier(text, candidate_labels=["Python Developer", "Data Scientist", "AI Engineer", "Frontend Developer", "Backend Developer"])
    print("Predicted role: ", result["labels"][0])
    print("Confidence: ", result["scores"][0])


if __name__ == "__main__":
    print("Basic AI resume Classifier")
    text = extract_text_from_pdf(r"sample_resume.pdf")
    resume_classifier(text=text[:3000])
