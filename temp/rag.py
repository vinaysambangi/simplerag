from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEXT_DIR = BASE_DIR / "data" / "text_files"
PDF_DIR = BASE_DIR / "data" / "pdf"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"

print(BASE_DIR)
print(TEXT_DIR)
print(PDF_DIR)
print(VECTOR_DIR)
injectionDone = False

# %%
###Data Injection
from langchain_core.documents import Document

# %%
doc = Document(
    page_content="this is the environment that make u more effective",
    metadata={
        "source":"the Book",
        "Author":"VNY Sambangi",
        "pages":3,
        "timestamp":"2026-08-04"
    }
)
doc

# %%
import os
os.makedirs("../data/text_files", exist_ok=True)

# %%
sample_texts = {"../data/text_files/lattice.txt":"""Unblock work and unlock potential with your personal AI Agent Support employees, managers, and leaders alike in doing their best work by proactively surfacing insights, answering questions about policies or career growth, and reinforcing positive habits.""",
                "../data/text_files/chuncking.txt":"""There are two big reasons why chunking is necessary for any application involving vector databases or LLMs: to ensure embedding models can fit the data into their context windows, and to ensure the chunks themselves contain the information necessary for search.
All embedding models have context windows, which determine the amount of information in tokens that can be processed into a single fixed size vector. Exceeding this context window may means the excess tokens are truncated, or thrown away, before being processed into a vector. This is potentially harmful
as important context could be removed from the representation of the text, which prevents it from being surfaced during a search."""}
for filepath,content in sample_texts.items():
    with open(filepath, "w", encoding='utf-8')as f:
        f.write(content)
print("sample text files were created")


# %%
from langchain_community.document_loaders import TextLoader

tl=TextLoader("../data/text_files/chuncking.txt", encoding='utf-8')
document =tl.load()
print(document)

# %%
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader,PyMuPDFLoader
text_loader = DirectoryLoader(
    str(TEXT_DIR),
    glob=["**/*.txt"],
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'},
    show_progress=False
)
print("Using PDF directory:", PDF_DIR)
pdf_loader=DirectoryLoader(
     str(PDF_DIR),
        glob=["**/*.pdf"],
        loader_cls=PyMuPDFLoader,
        show_progress=False
)
print("pdf_loader.path =", pdf_loader.path)
chuncks=text_loader.load() + pdf_loader.load() 
# for i in documents:
#     print(i.page_content)
chuncks

# %%
import numpy as np 
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# %%
class EmbeddigManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. EMbedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]):
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape : {embeddings.shape}")
        return embeddings
    
    #initialize the embedding manager
embedding_manager = EmbeddigManager()
embedding_manager 

# %%
class VectorStore:
    def __init__(self, collection_name : str = "Kamakshi", persist_directory: str = str(VECTOR_DIR)):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata= {"description":"PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store {e}")
            raise
        
    def add_documents(self, documents: List[Any], embeddings:np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")
        
        ids = []
        metadatas = []
        document_text = []
        embeddings_list = []
        
        for i,(doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            document_text.append(doc.page_content)
            
            embeddings_list.append(embedding.tolist())
            
        try:
            self.collection.add(
                ids=ids,
                embeddings = embeddings_list,
                metadatas = metadatas,
                documents = document_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"total documents in collection : {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
        
vector_store = VectorStore()
VectorStore

# %%
texts = [doc.page_content for doc in chuncks]
texts

# %%
#Convert the text to embeddings
texts = [doc.page_content for doc in chuncks]

#generate the embeddings
embeddings = (embedding_manager.generate_embeddings(texts))
vector_store.add_documents(chuncks, embeddings)

# %%
import os
print(os.getcwd())

# %%
class RAGRetriever:
    def __init__(self, vector_store : VectorStore, embedding_manager : EmbeddigManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
    def retrive(self, query: str, top_k:int = 5, score_threshold:float=0.0) -> List[Dict[str,Any]]:
        print(f"Retrive documents for query: '{query}'")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")
        
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrived_docs = []
            print(results.keys())
            print(results)
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i,(docid, document, metadata, distance) in enumerate (zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrived_docs.append({
                            'id' : docid,
                            'content' : document,
                            'metadata' : metadata,
                            'similarity_score' : similarity_score,
                            'distance' : distance,
                            'rank' : i+1
                        })
                print(f"Retrived {len(retrived_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrived_docs
        except Exception as e:
            print(f"Error during retrival : {e}")
            return[]
        
rag_retriever= RAGRetriever(vector_store, embedding_manager)

# %% [markdown]
# RAG Retrival

# %%
rag_retriever.retrive("Could u explain me the http onCOnnect")

# %% [markdown]
# Integration vector DB with the context pipeline with llm output

# %%
from openai import OpenAI, RateLimitError, APIError
# from clients import openai_client, CHAT_MODEL
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

openaiAPIkey = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openaiAPIkey, temperature=0.1, model='gpt-4o-mini', max_completion_tokens=1024)

#Rag function
def simpleRag(query, retriver, llm, top_k=5):
    results = rag_retriever.retriver(query, top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    
    prompt =  f"""You are a technical assistant helping RTI driver developers understand OEM/vendor API documentation for the device or system they are currently integrating.
                You will be given:              
                1. A developer's query              
                2. A set of retrieved documentation chunks (fetched via similarity search from a vector database of previously indexed API documentation)           
                RULES:          
                1. Answer ONLY using information contained in the provided context chunks below. Do not use prior knowledge of this API, this vendor, or similar APIs to fill gaps.       
                2. If the retrieved context does not contain enough information to answer the query, say so explicitly — do not guess, infer undocumented behavior, or hallucinate endpoints, parameters, request/response formats, error codes, or command syntax.           
                3. When you answer, cite which chunk(s) the information came from (e.g. "[Chunk 2]") so the developer can trace it back to the source document.          
                4. If multiple chunks contain conflicting or overlapping information (e.g. different versions of the same endpoint), point out the conflict rather than silently picking one.         
                5. Preserve exact technical details verbatim where precision matters — request/response JSON structures, command strings, hex/byte payloads, status/error codes, units, parameter names, and casing. Do not paraphrase or "clean up" these details in a way that changes them.     
                6. If the query is about implementation (e.g. "how do I structure the driver command for X"), answer at the level of what the documentation specifies (endpoints, payload structure, auth, polling/subscription model) — do not invent driver code unless the retrieved context includes actual reference driver code.           
                7. Keep responses concise and structured (use headers/bullets/code blocks) — developers are using this to quickly look up integration details, not read prose.      
                Context chunks:          
                {context}             
                Developer query:               
                {query}""" 
    response = llm.invoke([{"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": prompt}])
    return response.content

# %%
query = "How grouping is done with the denon device"

results = rag_retriever.retrieve(query)

answer = simpleRag(query, rag_retriever, llm)

print(answer)

# %%
