from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize models
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Initialize Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-1.5-flash")

# Load embeddings
print("Loading embeddings...")
def extract_vector(vector_str):
    try:
        # Clean the string and split by comma
        vector_str = vector_str.strip('[]')
        return np.array([float(x) for x in vector_str.split(',')])
    except Exception as e:
        return np.zeros(768)  # Return zero vector as fallback

def load_embeddings(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

# Load MedQuAD embeddings
medquad_embeddings = load_embeddings('../../biobert_processed_embeddings/train_medquad.csv')
if not medquad_embeddings.empty:
    medquad_embeddings['question_vector'] = medquad_embeddings['question_vector'].apply(extract_vector)

# Load all PubMed embeddings
pubmed_dfs = []
for i in range(1, 9):  # Assuming you have 5 embedding files
    file_path = f'../../biobert_processed_embeddings/processed_embeddings_{i}.csv'
    df = load_embeddings(file_path)
    if not df.empty:
        df['abstract_vector'] = df['abstract_vector'].apply(extract_vector)
        pubmed_dfs.append(df)

# Combine all PubMed embeddings
pubmed_embeddings = pd.concat(pubmed_dfs, ignore_index=True)
print(f"Total PubMed embeddings loaded: {len(pubmed_embeddings)}")

class ChatRequest(BaseModel):
    query: str

def get_biobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, 
                      truncation=True, padding=True)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def find_relevant_content(query):
    query_vector = get_biobert_embedding(query)
    
    # Find similar questions in MedQuad
    qa_scores = cosine_similarity([query_vector], np.vstack(medquad_embeddings['question_vector'].values))[0]
    top_qa_indices = np.argsort(qa_scores)[-3:][::-1]
    
    # Find similar abstracts in PubMed
    abstract_scores = cosine_similarity([query_vector], np.vstack(pubmed_embeddings['abstract_vector'].values))[0]
    top_abstract_indices = np.argsort(abstract_scores)[-3:][::-1]
    
    relevant_qa = medquad_embeddings.iloc[top_qa_indices]
    relevant_abstracts = pubmed_embeddings.iloc[top_abstract_indices]
    
    return relevant_qa, relevant_abstracts

def generate_response(query, relevant_qa, relevant_abstracts):
    prompt = f"You are a medical expert assistant. Answer the following medical question comprehensively and accurately. If the provided context contains relevant information, use it. If not, use your general medical knowledge to provide the best possible answer."

    prompt += f"\nQuestion: {query}\n"

    prompt += "\nContext:\n"
    prompt += "\nRelevant Medical QA Pairs:\n"
    for i, (_, row) in enumerate(relevant_qa.iterrows(), 1):
        prompt += f"\nQA Pair {i}:\nQ: {row['Question']}\nA: {row['Answer']}\n"
    
    prompt += "\nRelevant Medical Research Abstracts:\n"
    for i, (_, row) in enumerate(relevant_abstracts.iterrows(), 1):
        prompt += f"\nAbstract {i}:\n{row['abstract_text']}\n"
    
    prompt += "\nProvide a clear, direct, and comprehensive answer to the question. Focus on being helpful and informative to the user."
    
    response = llm_model.generate_content(prompt)
    return response.text

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        relevant_qa, relevant_abstracts = find_relevant_content(request.query)
        response = generate_response(request.query, relevant_qa, relevant_abstracts)
        return {"response": response, "status": "success"}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "status": "error"}

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)