import streamlit as st
from backend import get_answer, build_index

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 Your Personal Knowledge Chatbot")
st.write("Ask me anything based on your uploaded knowledge base (`blogs/`).")

try:
    build_index()
except Exception as e:
    st.warning(f"⚠️ Index build skipped: {e}")

user_query = st.text_input("🔍 Enter your question:")

if st.button("Get Answer") and user_query.strip():
    with st.spinner("🤔 Thinking..."):
        answer = get_answer(user_query)
    st.success("✅ Answer:")
    st.write(answer)
