from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from openai import OpenAI
import numpy as np
import pandas as pd
import fitz
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_words = [word for word in stemmed_words if not re.match(r'^#+', word)]
    stemmed_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in stemmed_words]
    processed_text = ' '.join(stemmed_words)
    return processed_text

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def compute_cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1).reshape(1, -1)
    vector2 = np.array(vector2).reshape(1, -1)
    cosine_sim = cosine_similarity(vector1, vector2)[0][0]
    return cosine_sim

def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def compute_matching_score(jd_keywords, resume_keywords):
    keyword_intersection = set(jd_keywords) & set(resume_keywords)
    score = len(keyword_intersection) / max(len(jd_keywords), len(resume_keywords))
    return score

@app.route('/process', methods=['POST'])
def process_resumes():
    jd = request.form['jd']
    idealresumes_text = ""
    for file in request.files.getlist('idealresumes'):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            idealresumes_text += '\n' + page.extract_text()
    
  
    embedIdeal = get_embedding(preprocess_text(idealresumes_text) + " ".join(jd.split('\n')))
    
    resumes = {}
    
    for file in request.files.getlist('resumes'):
        text = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += '\n' + page.extract_text()
        resumes[file.filename] = text
        
  

    resumes_df = pd.DataFrame(list(resumes.items()), columns=['Filename', 'Text'])
    resumes_df["embeddings"] = resumes_df["Text"].apply(lambda x: get_embedding(preprocess_text(x)))
    resumes_df["Text_Similarity"] = resumes_df["embeddings"].apply(lambda x: compute_cosine_similarity(x, embedIdeal))
    resumes_df["ats"] = resumes_df["Text"].apply(lambda x: compute_matching_score(extract_keywords(" ".join(jd.split('\n')) + idealresumes_text), extract_keywords(x)))

    Weights = [0.6,0.4]
    x = Weights[0]
    y = Weights[1]
    resumes_df["avg"] = (resumes_df["Text_Similarity"] * x + resumes_df["ats"] * y)
    sorted_resumes_df = resumes_df.sort_values(by="avg", ascending=False)

    
    
    sorted_resumes_df["Percentile"] = sorted_resumes_df['avg'].rank(pct=True)*100
    percentile = sorted_resumes_df.set_index('Filename')['Percentile'].to_dict()
    sorted_resumes_df['Rank'] = sorted_resumes_df['avg'].rank(ascending=False, method='min')
    rank = sorted_resumes_df.set_index('Filename')['Rank'].to_dict()
    sorted_resumes_df.drop(['Percentile', 'Rank'], axis=1)

    result = sorted_resumes_df[['Filename', 'Text_Similarity', 'ats', 'avg']].to_dict(orient='records')

    response = {
        'percentile': percentile,
        'rank': rank,
        'result': result

    }
    

    return jsonify(response)



if __name__ == '__main__':
    app.run(port=10000,debug=True)
