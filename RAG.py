import os, tempfile
from pathlib import Path
from glob import glob

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# from langchain.llms import VertexAI
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
    DirectoryLoader,
)
from langchain.text_splitter import (
    CharacterTextSplitter,
    TextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.indexes import VectorstoreIndexCreator


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
pages = []
for path in glob("data/*"):
    loader = PyPDFLoader(path)
    pages += loader.load_and_split()


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(
    pages, embedding=embeddings, persist_directory="./shoumikdb"
)
vectordb.persist()

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = HuggingFaceHub(
    repo_id="declare-lab/flan-alpaca-large",
    # repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.1, "max_length": 256},
    huggingfacehub_api_token="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",
)
# llm = VertexAI(
#     model_name="text-bison",
#     temperature=0.1,
#     top_k=20,
#     top_p=0.95,
#     max_output_tokens=2000,
# )

rqa_prompt_template = """Use the following pieces of context to answer the question at the end. 
Answer only from the context. If you do not know the answer, say you do not know. Elaborate the answers whenever possible. 
{context}

Question: {question}
"""
RQA_PROMPT = PromptTemplate(
    template=rqa_prompt_template, input_variables=["context", "question"]
)

rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}

qa = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=rqa_chain_type_kwargs,
    return_source_documents=True,
    verbose=False,
)

query = "What are adapter techniques? How is its performance"
result = qa({"query": query})

print(result["result"])


# # Option 2: Conversational
chat_history = []
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever
)  # retriever is same as prev initialised in 1.1

query = "What are adapter techniques??"
result = qa({"question": query, "chat_history": chat_history})
print(query)
print(result["answer"])
print()

chat_history = [(query, result["answer"])]
query = "How is its performance?"
result = qa({"question": query, "chat_history": chat_history})
print(query)
print(result["answer"])
print()
