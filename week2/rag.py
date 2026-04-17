import os
os.environ.setdefault("USER_AGENT", "rag-bootcamp")
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

PDF_PATH = os.path.join(DATA_DIR, "thinkpython2.pdf")
WEB_URL = "https://realpython.com/ref/"
PEP_FILES = [
    os.path.join(DATA_DIR, "pep-0008.txt"),
    os.path.join(DATA_DIR, "pep-0020.txt"),
    os.path.join(DATA_DIR, "pep-0257.txt"),
]

PROMPT_TEMPLATE = """Use only the context below to answer the question. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:"""


def load_all_documents():
    docs = []

    pdf_loader = PyPDFLoader(PDF_PATH)
    pdf_docs = pdf_loader.load()
    for d in pdf_docs:
        d.metadata["source_type"] = "pdf"
        d.metadata["source_name"] = "Think Python 2"
    docs.extend(pdf_docs)

    web_loader = WebBaseLoader(WEB_URL)
    web_docs = web_loader.load()
    for d in web_docs:
        d.metadata["source_type"] = "web"
        d.metadata["source_name"] = "Real Python Ref"
    docs.extend(web_docs)

    for path in PEP_FILES:
        pep_loader = TextLoader(path)
        pep_docs = pep_loader.load()
        pep_name = os.path.basename(path).replace(".txt", "").upper()
        for d in pep_docs:
            d.metadata["source_type"] = "pep"
            d.metadata["source_name"] = pep_name
        docs.extend(pep_docs)

    return docs


def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, persist=True):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    kwargs = {"persist_directory": CHROMA_DIR} if persist else {}
    return Chroma.from_documents(chunks, embeddings, **kwargs)


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def create_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
