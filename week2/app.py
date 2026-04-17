import streamlit as st
from rag import load_vectorstore, create_chain

st.set_page_config(page_title="Python Knowledge Q&A", page_icon="🐍")
st.title("🐍 Python Knowledge Base Q&A")
st.caption("Sources: Think Python 2 (PDF) · Real Python Ref (Web) · PEP 8, 20, 257 (Text)")


@st.cache_resource
def get_chain():
    vs = load_vectorstore()
    return create_chain(vs)


try:
    chain = get_chain()
except Exception as e:
    st.error(f"Failed to load vector store: {e}\nRun the notebook first to build the index.")
    st.stop()

query = st.text_input("Ask a question about Python:", placeholder="e.g. What is a list comprehension?")

if query:
    with st.spinner("Searching knowledge base..."):
        result = chain.invoke({"query": query})

    st.markdown("### Answer")
    st.write(result["result"])

    with st.expander("📄 Retrieved source chunks"):
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source_name", "unknown")
            stype = doc.metadata.get("source_type", "")
            st.markdown(f"**Chunk {i} — {source} ({stype})**")
            st.text(doc.page_content[:300] + "...")
            st.divider()
