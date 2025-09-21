import streamlit as st
from retriever_pipeline import RAGRetrieverPipeline
from vectorstore import VectorStore
from embedder import Embedder

# --- Data ingestion step ---
if "vectorstore" not in st.session_state or "embedder" not in st.session_state:
    from data_ingestion import main as run_data_ingestion
    vectorstore, embedder = run_data_ingestion()
    st.session_state.vectorstore = vectorstore
    st.session_state.embedder = embedder
else:
    vectorstore = st.session_state.vectorstore
    embedder = st.session_state.embedder


# vectorstore = VectorStore()
# embedder = Embedder()
pipeline = RAGRetrieverPipeline(vectorstore=vectorstore, embedder=embedder)

from llm_method import get_llm_response


st.title("GenAI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history above input
st.markdown("### Chat History")
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")

user_input = st.text_input("Ask a question:")
# if "user_input" not in st.session_state:
st.session_state.user_input = user_input

if st.button("Search"):
    if user_input:
        results = pipeline.answer_query(st.session_state.user_input, top_k=3)
        if results:
            response = ""
            for i, res in enumerate(results):
                content   = res.get('content', '')
                content   = get_llm_response(st.session_state.user_input, content)
                metadata  = res.get('metadata', {})
                location  = metadata.get('source', 'Unknown location')
                id        = res.get('id', 'N/A')  # Get slide number/id
                score     = res.get('similarity_score', 'N/A')  # Show N/A if score missing
                response += f"**Context {i+1}:**\n"
                response += f"- **Document Location:** {location}\n"
                response += f"- **Slide Number:** {id}\n"
                response += f"- **Match Score:** {score if score != 'N/A' else 'N/A'}\n"
                response += f"- **Content:** {content}\n\n"
        else:
            response = "Sorry, I couldn't find a relevant answer."
        st.session_state.chat_history.append(("You", st.session_state.user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.session_state.user_input = ""  # Clear input after search

# (Chat history is already shown above input)