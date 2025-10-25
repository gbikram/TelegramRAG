import os
from langchain_community.document_loaders import TelegramChatApiLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Change models here
local_embeddings = OllamaEmbeddings(model="all-minilm")
llm = OllamaLLM(model="deepseek-r1:1.5b")

telegram_channel_id = input("Telegram Channel ID: ")

loader = TelegramChatApiLoader(
    chat_entity=f"t.me/{telegram_channel_id}",  # recommended to use Entity here
    api_hash="",
    api_id="",
    username=telegram_channel_id,  # needed only for caching the session.
)

telegram_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)
all_splits = text_splitter.split_documents(telegram_docs)


# Generate vectorstore from docs if vectorstore dir is empty, otherwise load saved vectordb
# Split the documents into smaller batches (https://community.openai.com/t/error-while-reading-pdf-file-using-openai-chromadb-module/883612)
batch_size = 5461  # Set to the maximum allowed batch size
for i in range(0, len(all_splits), batch_size):
    batch = all_splits[i:i + batch_size]
    vectorstore = Chroma.from_documents(
    	documents=filter_complex_metadata(batch), 
    	embedding=local_embeddings, 
    	persist_directory="vectordb"
	)

while True:
    question = input("Question: ")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    context = ' '.join([doc.page_content for doc in retrieved_docs])
    response = llm.invoke(f"""
	    Answer the question according to the context:
    	    Question: {question}
    	    Context: {context}"""
    )
    print(response)
