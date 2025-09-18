import streamlit as st
from retriever_pipeline import RAGRetrieverPipeline
from vectorstore import VectorStore
from embedder import Embedder

# Initialize pipeline (make sure your vectorstore is already populated)
vectorstore = VectorStore()
embedder = Embedder()
pipeline = RAGRetrieverPipeline(vectorstore=vectorstore, embedder=embedder)

st.title("GenAI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")

if st.button("Send") and user_input:
    results = pipeline.answer_query(user_input, top_k=3)
    if results:
        answer = results[0]['content']
    else:
        answer = "Sorry, I couldn't find a relevant answer."
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")