# rag_engine.py
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGEngine:
    def __init__(self, docs_path):
        self.docs_path = docs_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Light embedding model
        self.docs, self.doc_names = self.load_documents()
        self.index = self.create_index()

    def load_documents(self):
        docs = []
        names = []
        for file in os.listdir(self.docs_path):
            if file.endswith('.txt'):
                path = os.path.join(self.docs_path, file)
                with open(path, 'r', encoding='utf-8') as f:
                    docs.append(f.read())
                    names.append(file.replace('.txt', '').lower())
        return docs, names

    def create_index(self):
        embeddings = self.model.encode(self.docs)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        return index

    def search(self, query):
        query_vec = self.model.encode([query])
        _, I = self.index.search(np.array(query_vec), k=1)
        return self.docs[I[0][0]]

    def generate_report(self, patient_name, disease_name):
        if disease_name.lower() == "no disorder":
            return f"✅ Patient {patient_name} shows no signs of a sleep disorder."

        disease_info = self.search(disease_name)
        report = f"""🧾 Patient Report: {patient_name}

Detected Sleep Disorder: {disease_name}

Details:
{disease_info}

Note: Please consult a licensed doctor for personalized diagnosis and treatment.
"""
        return report
