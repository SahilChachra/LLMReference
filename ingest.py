# Character splitters used to split text data in your pdf/csv/etc
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# Vector Store/Databases
from langchain_community.vectorstores import FAISS, Qdrant

# Embeddings creators
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Document Loaders
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader

class FAISSVectorStores:

    def __init__(self, 
            dataset_path="./data", 
            dataset_type="pdf", 
            text_splitter="recursive", 
            embeddings_model='sentence-transformers/all-MiniLM-L6-v2', 
            faiss_db_path="./faiss_vector_store",
            chunk_size=500,
            chunk_overlap=50,
            device="cuda",
            use_multithreading=True):

        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.text_splitter = text_splitter
        self.embeddings_model = embeddings_model
        self.faiss_db_path = faiss_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.use_multithreading = use_multithreading

    def create_vector_store(self):
        if self.dataset_type == "pdf":
            loader = DirectoryLoader(self.dataset_path, glob='*.pdf', 
                loader_cls=PyPDFLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)
        elif self.dataset_type == "csv":
            loader = DirectoryLoader(self.dataset_path, glob='*.csv', 
                loader_cls=CSVLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)
        elif self.dataset_type == "md":
            loader = DirectoryLoader(self.dataset_path, glob='*.md',
                show_progress=True, use_multithreading=self.use_multithreading)
        else:
            loader = DirectoryLoader(self.dataset_path, glob='*.*', 
                loader_cls=UnstructuredFileLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)

        documents = loader.load()

        if self.text_splitter == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                chunk_overlap = self.chunk_overlap)
        else:
            text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size,
                chunk_overlap = self.chunk_overlap)

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model, 
                        model_kwargs={'device' : self.device})

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.faiss_db_path)


class QdrantVectorDatabase:
    '''
    Steps to setup :-
        1. docker pull qdrant/qdrant
        2. docker run -p 6333:6333 -p 6334:6334 -v {PATH_TO_YOUR_DIRECTORY}:/qdrant/storage:z qdrant/qdrant
    '''
    def __init__(self, 
            dataset_path="./data", 
            dataset_type="pdf", 
            text_splitter="recursive", 
            embeddings_model='sentence-transformers/all-MiniLM-L6-v2', 
            faiss_db_path="./faiss_vector_store",
            chunk_size=500,
            chunk_overlap=50,
            device="cuda",
            qdrant_url = "http://localhost:6333",
            name_of_db = "sample_vector_db",
            use_multithreading=True):

        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.text_splitter = text_splitter
        self.embeddings_model = embeddings_model
        self.faiss_db_path = faiss_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.qdrant_url = qdrant_url
        self.collection_name = name_of_db
        self.use_multithreading = use_multithreading
    
    def create_vector_database(self):
        if self.dataset_type == "pdf":
            loader = DirectoryLoader(self.dataset_path, glob='*.pdf', 
                loader_cls=PyPDFLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)
        elif self.dataset_type == "csv":
            loader = DirectoryLoader(self.dataset_path, glob='*.csv', 
                loader_cls=CSVLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)
        elif self.dataset_type == "md":
            loader = DirectoryLoader(self.dataset_path, glob='*.md',
                show_progress=True, use_multithreading=self.use_multithreading)
        else:
            loader = DirectoryLoader(self.dataset_path, glob='*.*', 
                loader_cls=UnstructuredFileLoader, show_progress=True, 
                use_multithreading=self.use_multithreading)

        documents = loader.load()

        if self.text_splitter == "recursive":
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                chunk_overlap = self.chunk_overlap)
        else:
            text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size,
                chunk_overlap = self.chunk_overlap)

        texts = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name=self.embeddings_model)
        
        try:
            qdrant = Qdrant.from_documents(
                texts,
                embeddings,
                url = self.qdrant_url,
                prefer_grpc = False,
                collection_name = self.collection_name
            )
            print("Vectors created!")
        except Exception as e:
            print("Counldn't create Qdrant Database!")
            print("Exception : ", e)
            return 0
        

if __name__ == "__main__":

    # Create object of class FAISS Vector Store
    # faiss_vector_store = FAISSVectorStores()
    # Call this function to create and store embeddings
    # faiss_vector_store.create_vector_store()

    # Create object of Qdrant DB class
    qdrant_vector_database = QdrantVectorDatabase()
    # Call this function to create and store embeddings
    qdrant_vector_database.create_vector_database()