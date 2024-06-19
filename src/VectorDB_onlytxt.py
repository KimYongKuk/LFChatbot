from Constants import *
import chromadb
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

## VectorDB : ChromaDB
## EmbeddingModel : klue-sroberta-base

# region Vector, Embedding 변수 정의
#client 생성 (PersistentClient의 경우, 영구적으로 디스크에 저장 가능)
perClient = chromadb.PersistentClient(
    path=CHROMADB_PATH
)

#Collection 생성
posts = perClient.get_or_create_collection (
    name=COLLECTION_NAME,
    # metadata={'hnsw:space': 'cosine'},
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
)

#임베딩 모델, dimension 768
embeddingModel = SentenceTransformerEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")

#set vectorStore
vectorStore = Chroma(
    client = perClient,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddingModel
)
# endregion

# region 함수 정의

#return answer in vectorDB
def answer_vectorDB(query): # query = "경영정보팀에 대해 알려줘"
    docs = vectorStore.similarity_search(query)
    return docs

#split text
def split_value():
    loader = DirectoryLoader('./data', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = 20)
    texts = text_splitter.split_documents(documents)
    return texts

#text파일을 임베딩 csv로 변환하는 함수
def create_embedding():
    #split 문서정보를 Collection에 add.
    checkValue = 's'
    try:
        tmpIdx = 0
        for doc in split_value():
            posts.upsert(
                # ids=['문서' + str(tmpIdx)], 
                ids=[doc.metadata["source"]], 
                documents=doc.page_content
                # metadatas=[{"doc_id" : i for i in range (1000)}]
            )
            tmpIdx += 1
    except Exception as e:
        checkValue = str(e)
        
    print(vectorStore._collection.count(), "in the collection")
    return checkValue
# endregion
    
