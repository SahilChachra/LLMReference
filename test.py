from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompt_data import CustomPromptData
from prompt_template import CustomPromptTemplate
from model import LLMModel
from chains import ConversationalRetrievalQAChain
 
if __name__ == "__main__":

    # Load the model
    llmModel = LLMModel()
    llm = llmModel.getLLM()

    # Create a prompt template with chat_history and current question
    condense_prompt_data = """
        The current chat history is {chat_history}.

        User's current question is {question}
    """
    # Create Prompt template obejct for it
    condense_prompt = CustomPromptTemplate(CustomPromptData(condense_prompt_data).get_prompt_data(), 
            input_variables=["question", "chat_history"]).getCustomPromptTemplate()

    # Content for prompt template which will take in retrieved output from data store and current question
    combine_custom_data = """You are an AI assistant who helps the user with their questions. Given the context below, answer the user's question. If you don't know the answer, politely tell that you are unware of that and do not try to make an answer of your own.

        Context: {context}
        Question: {question}

        Return the answer below.
        Answer:
        """
    # Create prompt template object
    combine_custom_prompt = CustomPromptTemplate(CustomPromptData(combine_custom_data).get_prompt_data(), 
            input_variables=["context", "question"]).getCustomPromptTemplate()

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    # Load FAISS vector store
    db = FAISS.load_local("faiss_vector_store", embeddings)

    # Create Retrieval QA Chain object and fetch the chain
    convRetrievalQAChainObject = ConversationalRetrievalQAChain(llm=llm, condense_prompt=condense_prompt, 
                                       combine_docs_custom_prompt= combine_custom_prompt, db=db)
    chain, memory = convRetrievalQAChainObject.getConversationalRetrievalQAChain()
    
    # Run inference
    ques_ = "When was pytorch developed?"
    print(f"Question : {ques_}\nAnswer : {chain.invoke(ques_)['answer']}")
    print(f"\n Memory buffer : {memory.buffer}  \n")
    
    ques_ = "What is the latest version of PyTorch?"
    print(f"Question : {ques_}\nAnswer : {chain.invoke(ques_)['answer']}")
    print(f"\n Memory buffer : {memory.buffer} \n")
