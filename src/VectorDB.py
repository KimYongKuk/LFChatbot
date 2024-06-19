from Constants import *
import chromadb
import os
import uuid

from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Settings

class VectorDB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorDB, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True

            ## VectorDB : ChromaDB
            ## EmbeddingModel : klue-sroberta-base

            # region Vector, Embedding 변수 정의
            #client 생성 (PersistentClient의 경우, 영구적으로 디스크에 저장 가능)
            self.perClient = chromadb.PersistentClient(
                path=CHROMADB_PATH, 
                settings=Settings(allow_reset=True)
            )

            #임베딩 모델, dimension 768
            self.embeddingModel = SentenceTransformerEmbeddings(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")    

            #set vectorStore
            self.vectorStore = Chroma(
                client = self.perClient,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddingModel
            )

            #Collection 생성
            self.posts = self.perClient.get_or_create_collection (
                name=COLLECTION_NAME,
                # metadata={'hnsw:space': 'cosine'},
                embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr")
            )

    # endregion

    # region 함수 정의

    #return answer in vectorDB
    def answer_vectorDB(self, query): # query = "경영정보팀에 대해 알려줘"
        # vector_db = VectorDB()
        docs = self.vectorStore.similarity_search(query)
        return docs

    #split text
    def split_value(self):
        txtLoader = DirectoryLoader(DOC_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
        pdfLoader = DirectoryLoader(DOC_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        
        all_documents = txtLoader.load() + pdfLoader.load()
        # loader_all = MergedDataLoader(loaders=[txtLoader, pdfLoader])

        # os.remove("./data")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = 20)
        #documents = loader_all.load_and_split(text_splitter=text_splitter)
        texts = text_splitter.split_documents(all_documents)
        return texts

    #text파일을 임베딩 csv로 변환하는 함수
    def create_embedding(self):

        #split 문서정보를 Collection에 add.
        checkValue = 's'

        try:
            tmpIdx = 0
            for doc in self.split_value():

                print(doc)
                # print(f"Processing document {tmpIdx}: {doc}")
                # print(f"Document {tmpIdx} content: {doc.page_content}")
                self.posts.add(
                    # ids=['문서' + str(tmpIdx)], 
                    # ids=[doc.metadata["source"]], 
                    ids = [str(uuid.uuid1())],
                    # uris= doc.metadata["source"],
                    # metadatas={"source": doc.metadata["source"]},
                    documents=doc.page_content
                    # metadatas=[{"doc_id" : i for i in range (1000)}]
                )
                # print(doc.page_content)
                tmpIdx += 1
        except Exception as e:
            checkValue = str(e)
        return checkValue
    # endregion


    
