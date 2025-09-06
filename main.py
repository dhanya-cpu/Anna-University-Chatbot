import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from transformers import pipeline
urls = [
    "https://www.annauniv.edu/",
    "https://www.annauniv.edu/study/programmes.php",
    "https://www.annauniv.edu/contact.php"
]
docs = []
for url in urls:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
st.title("Anna University Chatbot 🤖")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
query = st.text_input("Ask me anything about Anna University:")
if query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    results = retriever.get_relevant_documents(query)
    if results:
        context = " ".join([res.page_content for res in results])
        prompt = f"Answer the following question based only on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
        answer = qa_pipeline(prompt, max_new_tokens=200)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer[0]['generated_text']))
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
