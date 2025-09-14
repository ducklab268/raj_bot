# 📚 Python RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions using custom text content like blog posts.  
Built with **FAISS**, **LangChain**, and a Hugging Face LLM with a simple **Streamlit** interface.

---

## 🚀 Features
- Load and chunk text files from the `blogs/` folder
- Generate embeddings using `sentence-transformers`
- Store embeddings in **FAISS** for fast retrieval
- Use **LangChain**'s `ConversationalRetrievalChain` to answer queries
- Streamlit frontend with chat history for interactive Q&A

---

## 📁 Project Structure

rag_test/
├── blogs/
│ ├── blog1.txt
│ └── blog2.txt
├── rag_pipeline.ipynb # Main notebook for RAG logic
├── app.py # Streamlit frontend
└── requirements.txt # Dependencies


---

## ⚡ Setup & Usage

1. Clone the repository:
```bash
git clone https://github.com/ducklab268/rag-chatbot.git
cd rag-chatbot

pip install -r requirements.txt
jupyter notebook rag_pipeline.ipynb
streamlit run app.py

💬 Example Queries
“What is energy mastery?”
“How do high performers avoid burnout?”
Tools & Libraries
Python 3.x
LangChain



✨ Author

Mirza Takreem Ahmed
