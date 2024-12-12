from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import time

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ChatRequest(BaseModel):
    query: str
    chatId: Optional[str] = None

print("Loading models and embeddings...")

# Load environment variables
load_dotenv()

# Initialize models - do this only once
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Initialize Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-1.5-flash")

# Global variables to store embeddings
medquad_embeddings = None
pubmed_embeddings = None

def extract_vector(vector_str):
    try:
        vector_str = vector_str.strip('[]')
        return np.array([float(x) for x in vector_str.split(',')])
    except Exception as e:
        return np.zeros(768)

def load_embeddings():
    global medquad_embeddings, pubmed_embeddings
    
    # Load MedQuAD embeddings if not already loaded
    if medquad_embeddings is None:
        print("Loading MedQuAD embeddings...")
        df = pd.read_csv('../../biobert_processed_embeddings/train_medquad.csv')
        df['question_vector'] = df['question_vector'].apply(extract_vector)
        medquad_embeddings = df
    
    # Load PubMed embeddings if not already loaded
    if pubmed_embeddings is None:
        print("Loading PubMed embeddings...")
        pubmed_dfs = []
        for i in range(1, 9):
            file_path = f'../../biobert_processed_embeddings/processed_embeddings_{i}.csv'
            try:
                df = pd.read_csv(file_path)
                df['abstract_vector'] = df['abstract_vector'].apply(extract_vector)
                pubmed_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        pubmed_embeddings = pd.concat(pubmed_dfs, ignore_index=True)
        print(f"Total PubMed embeddings loaded: {len(pubmed_embeddings)}")

# Load embeddings at startup
load_embeddings()

@torch.no_grad()
def get_biobert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, 
                      truncation=True, padding=True)
    outputs = biobert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def find_relevant_content(query):
    query_vector = get_biobert_embedding(query)
    
    # Find similar questions in MedQuad
    qa_scores = cosine_similarity([query_vector], np.vstack(medquad_embeddings['question_vector'].values))[0]
    top_qa_indices = np.argsort(qa_scores)[-3:][::-1]
    
    # Find similar abstracts in PubMed
    abstract_scores = cosine_similarity([query_vector], np.vstack(pubmed_embeddings['abstract_vector'].values))[0]
    top_abstract_indices = np.argsort(abstract_scores)[-3:][::-1]
    
    return medquad_embeddings.iloc[top_qa_indices], pubmed_embeddings.iloc[top_abstract_indices]

def generate_response(query, relevant_qa, relevant_abstracts):
    prompt = f"""You are a medical expert assistant. Answer the following medical question comprehensively and accurately. 
    If the provided context contains relevant information, use it. If not, use your general medical knowledge to provide the best possible answer.
    
    Question: {query}
    
    Context:
    Relevant Medical QA Pairs:
    """
    
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
        chat_id = request.chatId if request.chatId else str(int(time.time() * 1000))
        
        # Generate response
        relevant_qa, relevant_abstracts = find_relevant_content(request.query)
        response = generate_response(request.query, relevant_qa, relevant_abstracts)
        
        timestamp = datetime.now().isoformat()
        
        # Create messages
        user_message = {
            "role": "user",
            "content": request.query,
            "timestamp": timestamp,
            "chatId": chat_id
        }
        
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": timestamp,
            "chatId": chat_id
        }
        
        # Update history
        history = load_chat_history()
        if "messages" not in history:
            history["messages"] = []
        
        history["messages"].append(user_message)
        history["messages"].append(assistant_message)
        save_chat_history(history)
        
        return {
            "response": response,
            "status": "success",
            "timestamp": timestamp,
            "chatId": chat_id
        }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"response": str(e), "status": "error"}

def load_chat_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"messages": []}
    except json.JSONDecodeError:
        return {"messages": []}

def save_chat_history(history):
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=2)

@app.get("/chat-history")
async def get_chat_history():
    try:
        history = load_chat_history()
        return {
            "history": history.get("messages", []),
            "status": "success"
        }
    except Exception as e:
        print(f"Error getting chat history: {str(e)}")
        return {
            "history": [],
            "status": "error",
            "message": str(e)
        }

@app.post("/clear-history")
async def clear_history():
    try:
        save_chat_history({"messages": []})
        return {"status": "success"}
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)