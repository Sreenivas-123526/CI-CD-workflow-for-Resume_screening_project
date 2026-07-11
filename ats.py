import fitz  # PyMuPDF
import docx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Common technical skills
SKILLS = [
    "python", "java", "sql", "machine learning",
    "deep learning", "tensorflow", "pytorch",
    "nlp", "streamlit", "flask", "django",
    "aws", "azure", "docker", "kubernetes",
    "pandas", "numpy", "scikit-learn",
    "git", "linux", "power bi", "tableau"
]

def extract_text(file):

    if file.name.endswith(".pdf"):

        pdf = fitz.open(stream=file.read(), filetype="pdf")

        text = ""

        for page in pdf:

            text += page.get_text()

        return text

    elif file.name.endswith(".docx"):

        document = docx.Document(file)

        return "\n".join([p.text for p in document.paragraphs])

    elif file.name.endswith(".txt"):

        return file.read().decode("utf-8")

    return ""


def calculate_ats_score(resume, jd):

    cv = CountVectorizer()

    matrix = cv.fit_transform([resume, jd])

    score = cosine_similarity(matrix)[0][1]

    return score * 100


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        if skill in text:

            found.append(skill)

    return list(set(found))


def missing_skills(resume_skills, jd_skills):

    return list(set(jd_skills) - set(resume_skills))