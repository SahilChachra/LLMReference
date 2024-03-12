What does Ingest.py does?

    While working with RAG, you have to generate some embeddings or data about the task you are going to perform.
    You may be working on some task on which the model is not trained or the training data is older.
    So what you do is, take your data (maybe PDF/CSV/Text files) and generate embeddings out of it.
    These embeddings are then stored in a vector store or database to help you fetch it whenever required.
    So, ingest.py helps you generate and store embeddings in given database/data store.

    If reads your data, split it into smaller parts, generate embeddings and store them.

What are Vector Stores?

What are Vector Databases?

Vector vs Traditional Database

What are some examples of Vector Stores?

    FAISS
    Chroma

What are embeddings models?

What is Chunk size and Chunk overlap?

    Chunk size - While generating embeddings, how many tokens will constitute to 1 block/document

    Chunk overlap - When the text is too big (more than chunk size, technically), we divide it into different documents/blocks.
                    Chunk overlap denotes how many tokens to overlap with the previous block/document to maintain continuity.

Few famous Embedding models?

What is AutoModelForCausalLM?

What is Llamacpp?

    Llamacpp is a library meant to help devs load various models and run them on various hardwares.
    It focuses on optimization - speed and performance

    Parameters while loading a model

    model_path : path to the model
    temperature : decides how deterministic or creative the output as to be. 0.1 - deterministic/conservative output and 0.9 - more creative and different output
    max_tokens : maximum num of tokens to generate
    top_p : The model outputs a set of next tokens and their probabilities. Top_p selects a set of token such that their      probability scores sum up to 1. Then choose 1 token from it and return a output
    n_gpu_layers : num of layers to load in GPU
    n_batch : number of tokens to process in parallel. Should be less than max_tokens
    verbose : print the info while loading the model

What is CTransformers?

What is GGUF and GGML?

How does caching the LLM model helps?

What are different types of Memory available?
    
    ConversationBufferMemory
    ConversationBufferWindowMemory
    ConversationSummaryMemory
    ConversationSummaryBufferMemory

    // https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/