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
def extract_vector(vector_str, delimiter=None):
    try:
        # Clean the string and split by specified delimiter or whitespace
        vector_str = vector_str.strip('[]')
        if delimiter:
            return np.array([float(x) for x in vector_str.split(delimiter)])
        return np.array([float(x) for x in vector_str.split()])
    except Exception as e:
        print(f"Error extracting vector: {str(e)}")
        return np.zeros(768)  # Return zero vector as fallback

def load_embeddings(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

# Load MedQuAD embeddings (using space delimiter)
medquad_embeddings = load_embeddings('../../biobert_processed_embeddings/train_medquad.csv')
if not medquad_embeddings.empty:
    medquad_embeddings['question_vector'] = medquad_embeddings['question_vector'].apply(
        lambda x: extract_vector(x)  # Default delimiter (space)
    )

# Load all PubMed embeddings (using comma delimiter)
pubmed_dfs = []
for i in range(1, 9):
    file_path = f'../../biobert_processed_embeddings/processed_embeddings_{i}.csv'
    df = load_embeddings(file_path)
    if not df.empty:
        df['abstract_vector'] = df['abstract_vector'].apply(
            lambda x: extract_vector(x, delimiter=',')  # Comma delimiter
        )
        pubmed_dfs.append(df)

# Combine all PubMed embeddings
pubmed_embeddings = pd.concat(pubmed_dfs, ignore_index=True)
print(f"Total PubMed embeddings loaded: {len(pubmed_embeddings)}")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    status: str
    qa_pairs: list
    abstracts: list

def get_biobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, 
                      truncation=True, padding=True)
    with torch.no_grad():
        outputs = biobert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def find_top_k_abstracts(df, query_vector, abstract_vectors_col, k=3):
    cosine_scores = cosine_similarity([query_vector], np.array(df[abstract_vectors_col]).tolist())[0]
    top_k_indices = np.argsort(cosine_scores)[-k:][::-1]
    return df.iloc[top_k_indices]['abstract_id'].values, cosine_scores[top_k_indices]

def find_top_k_answers(df, query_vector, question_vectors_col, k=3):
    cosine_scores = cosine_similarity([query_vector], np.array(df[question_vectors_col]).tolist())[0]
    top_k_indices = np.argsort(cosine_scores)[-k:][::-1]
    return df.iloc[top_k_indices]['qa_id'].values, cosine_scores[top_k_indices]

def get_qa_pairs(df, qa_ids):
    return df[df['qa_id'].isin(qa_ids)][['qa_id', 'Question', 'Answer']]

def get_abstracts(df, abstract_ids):
    return df[df['abstract_id'].isin(abstract_ids)][['abstract_id', 'abstract_text']]

def query_top_k_answers_and_abstracts(df_1, df_2, query, k=10):
    query_vector = get_sentence_vector(query)

    qa_ids, question_scores = find_top_k_answers(df_1, query_vector, 'question_vector', k)
    abstract_ids, abstract_scores = find_top_k_abstracts(df_2, query_vector, 'abstract_vector', k)

    answers_df = get_qa_pairs(df_1, qa_ids)
    abstracts_df = get_abstracts(df_2, abstract_ids)

    answers_df['Similarity Score'] = question_scores
    abstracts_df['Similarity Score'] = abstract_scores

    return answers_df, abstracts_df

def find_relevant_content(query):
    query_vector = get_biobert_embedding(query)
    
    # Find similar questions in MedQuad (top 3)
    qa_ids, question_scores = find_top_k_answers(
        medquad_embeddings, 
        query_vector, 
        'question_vector', 
        k=3
    )
    
    # Find similar abstracts in PubMed (top 3)
    abstract_ids, abstract_scores = find_top_k_abstracts(
        pubmed_embeddings,
        query_vector,
        'abstract_vector',
        k=3
    )
    
    # Get the actual QA pairs and abstracts
    relevant_qa = get_qa_pairs(medquad_embeddings, qa_ids)
    relevant_abstracts = get_abstracts(pubmed_embeddings, abstract_ids)
    
    # Add similarity scores
    relevant_qa['Similarity Score'] = question_scores
    relevant_abstracts['Similarity Score'] = abstract_scores
    
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
        # Get relevant content
        relevant_qa, relevant_abstracts = find_relevant_content(request.query)
        
        # Generate LLM response
        response = generate_response(request.query, relevant_qa, relevant_abstracts)
        
        # Format QA pairs and abstracts for response
        qa_pairs = relevant_qa[['Question', 'Answer']].to_dict('records')
        abstracts = relevant_abstracts[['abstract_text']].to_dict('records')
        
        return ChatResponse(
            response=response,
            status="success",
            qa_pairs=qa_pairs,
            abstracts=abstracts
        )
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            status="error",
            qa_pairs=[],
            abstracts=[]
        )

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)