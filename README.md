# RAG pipeline using LangChain

A simple retrieval-augmented generation framework using LangChain.

For embeddings, I use the **all-mpnet-base-v2** model from HuggingFace.
For the knowledge base I use Chromadb, which is a vector management library. It is light weight and an easy alternative for vector databases (for small prototyping or dev projects).
For the LLM, I use **declare-lab/flan-alpaca-large*** from HuggingFace.


To run:

1. Install requirements using <br>
`pip install -r requirements.txt`

2. Create your knoweldge base by adding pdfs to the data folder. <br>

3. Run the code using <br>
`python RAG.py`



Some example questions you can try:

Search Mode <br>
1) What tools can I user for orchestration?
2) What is the LLaMA 2 Community License Agreement?
3) What vector databses can I use? 
4) What are some traditional ETL tools?
5) What do I need caching in LLMs?
