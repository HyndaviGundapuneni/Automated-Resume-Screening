# Automated Resume Screening | Artificial Intelligence

This project implements an **AI-based Automated Resume Screening System** using **Natural Language Processing (NLP)** techniques. It leverages **Named Entity Recognition (NER)**, **Text Classification**, **TF-IDF**, and **transformer models** to extract key information from resumes and match them effectively to job descriptions. This solution aims to enhance the recruitment process by ranking candidates based on their relevance to the job profile.

---

## Features

- **Named Entity Recognition (NER):** 
  Extracts key information like skills, certifications, and experience from resumes.
  
- **Text Classification:**
  Ranks resumes based on their semantic similarity to job descriptions.

- **TF-IDF and Cosine Similarity:**
  Measures relevance between resumes and job descriptions.

- **Transformer-based Semantic Matching:**
  Advanced matching using pre-trained transformer models for improved accuracy.

- **Accuracy:** Achieved **86%** accuracy in ranking candidates correctly.

---

## Technologies Used

- **Python**
- **Natural Language Processing (NLP):** Spacy, NLTK
- **Machine Learning:** Scikit-learn
- **Deep Learning Models:** Hugging Face Transformers
- **Libraries:** Pandas, NumPy, Matplotlib

---

## Setup Instructions

### Prerequisites

1. Python 3.7 or above
2. Install required libraries:

```bash
pip install spacy sklearn pandas numpy transformers nltk
Download the Spacy model:
bash
Copy code
python -m spacy download en_core_web_sm
Running the Project
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/automated-resume-screening.git
cd automated-resume-screening
Add or update the sample resumes and job descriptions in the code.

Run the main script:

bash
Copy code
python resume_screening.py
View the extracted entities and ranked candidates in the terminal output.
Example Outputs
Extracted Entities:
plaintext
Copy code
Resume 1: {'Skills': ['Python', 'SQL', 'AWS'], 'Experience': ['5 years of experience'], 'Certifications': []}
Resume 2: {'Skills': ['NLP', 'transformers', 'Python', 'Google Cloud Platform'], 'Experience': [], 'Certifications': ['certified AI engineer']}
Candidate Ranking:
plaintext
Copy code
Ranking Candidates:
1. Resume 2: Similarity Score = 0.75
2. Resume 1: Similarity Score = 0.68
Future Enhancements
Incorporate more sophisticated NER models to improve entity extraction accuracy.
Add more domain-specific datasets for enhanced performance.
Implement a GUI for user-friendly interaction.
Integrate into ATS (Applicant Tracking Systems) for live deployment.
Contribution Guidelines
Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes and submit a pull request.
License
This project is licensed under the MIT License.

