import os, tempfile
from pathlib import Path
from glob import glob

from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
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

HUGGINGFACEHUB_API_KEY = os.environ["HUGGINGFACEHUB_API_TOKEN"]
if HUGGINGFACEHUB_API_KEY:
    print(True)


def initialize():
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)

    # Read Data and split docs
    pages = []
    for path in glob("data/*"):
        loader = PyPDFLoader(path)
        pages += loader.load_and_split()

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Initialize chromadb
    vectordb = Chroma.from_documents(
        pages, embedding=embeddings, persist_directory="./shoumikdb"
    )
    vectordb.persist()

    # Initialize retriever. Using cosine similarity and retrieving 3 best matches.
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Initiate llm generator. Temperature = 0 (Not creative). Temperature = 1 (Creative). Using
    llm = HuggingFaceHub(
        repo_id="declare-lab/flan-alpaca-large",
        # repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.2, "max_length": 512},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY,
    )

    # Using Palm2 on vertex.
    # llm = VertexAI(
    #     model_name="text-bison",
    #     temperature=0.1,
    #     top_k=20,
    #     top_p=0.95,
    #     max_output_tokens=2000,
    # )

    return embeddings, vectordb, retriever, llm


def search_mode(retriever, llm):
    print("In Search mode")
    rqa_prompt_template = """Use the following pieces of context to answer the question at the end. 
    Answer only from the context. If you do not know the answer, say you do not know. 
    {context}
    Explain in detail. 
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
    query = input("Ask a question: ")
    result = qa({"query": query})

    print(result["result"])
    print()
    # print(result)


def chat_mode(retriever, llm):
    print("In Chat mode")

    # # Option 2: Conversational

    chat_history = []
    # Should take `chat_history` and `question` as input variables.

    choice = True
    while choice:
        query = input("Ask a question or press q to quit \n")
        if query.lower() == "q":
            choice == False
            print("Thank you")
            exit()

        prompt = (
            "Combine the chat history and follow up question into "
            "a standalone question. Chat History: {chat_history}"
            "Follow up question: {query}"
        )
        prompt = PromptTemplate.from_template(query)
        # question_generator_chain = LLMChain(llm=llm, prompt=prompt)

        qa = ConversationalRetrievalChain.from_llm(
            llm, retriever, condense_question_prompt=prompt
        )  # retriever is same as prev initialised in 1.1

        # chain = ConversationalRetrievalChain(
        #     retriever=retriever,
        #     question_generator=question_generator_chain,
        # )

        result = qa({"question": query, "chat_history": chat_history})
        # result = chain({"question": query, "chat_history": chat_history})

        print(result["answer"])
        chat_history = [(query, result["answer"])]


# Give the user some context.
print("\n What would you like to do?")

# Set an initial value for choice other than the value for 'quit'.
choice = ""
first = True
# Start a loop that runs until the user enters the value for 'quit'.
while choice != " ":
    # Give all the choices in a series of print statements.
    print("\n[1] Search mode.[Single questions]")
    print("\n[2] Chat mode. [Asking a queston + follow up]")
    print("\n[q] Quit.")

    # Ask for the user's choice.
    choice = input("\nWhat would you like to do? ")

    # initialize knowledge base, retriever, and generator
    if first and choice != "q":
        print(f"Setting up knowledge base, retriever, and generator ")
        embeddings, vectordb, retriever, llm = initialize()
        first = False

    # Respond to the user's choice.
    if choice == "1":
        search_mode(retriever, llm)

    elif choice == "2":
        chat_mode(retriever, llm)

    elif choice == "q":
        print("\nThanks for playing. See you later.\n")
        exit()
    else:
        print("\nI don't understand that choice, please try again.\n")

# Print a message that we are all finished.
print("Thanks again, bye now.")
