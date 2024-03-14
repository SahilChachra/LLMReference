from langchain.chains import RetrievalQA
from model import LLMModel
from prompt_template import CustomPromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory


class RetrievalQAChain:
    def __init__(self, llm : LLMModel, prompt : CustomPromptTemplate, db : any, 
                chain_type : str, return_source_documents : bool) -> None:
        self.__llm = llm
        self.__prompt = prompt
        self.__db = db
        self.__chain_type = chain_type
        self.__return_source_documents = return_source_documents

    def getRetrievalQAChain(self) -> RetrievalQA:
        qna_chain = RetrievalQA.from_chain_type(llm=self.__llm,
                                            chain_type=self.__chain_type,
                                            retriever=self.__db.as_retriever(search_kwargs={'k': 2}),
                                            return_source_documents=self.__return_source_documents,
                                            chain_type_kwargs={'prompt': self.__prompt}
                                        )
        return qna_chain


class ConversationalRetrievalQAChain:
    def __init__(self, llm : LLMModel, condense_prompt : CustomPromptTemplate, 
                combine_docs_custom_prompt : CustomPromptTemplate, db : any) -> None:
        self.__llm = llm
        self.__db = db
        self.__condense_prompt = condense_prompt
        self.__combine_docs_custom_prompt = combine_docs_custom_prompt
    
    def getConversationalRetrievalQAChain(self) -> ConversationalRetrievalChain:
        memory = ConversationBufferMemory(memory_key="chat_history", 
            input_key="question", return_messages=True)

        conv_chain = ConversationalRetrievalChain.from_llm(
            self.__llm, 
            self.__db.as_retriever(search_kwargs={'k': 2}), # see below for vectorstore definition
            memory=memory,
            condense_question_prompt=self.__condense_prompt,
            combine_docs_chain_kwargs=dict(prompt=self.__combine_docs_custom_prompt)
        )

        return conv_chain, memory

class Conversation:
    def __init__(self, llm : LLMModel, prompt : CustomPromptTemplate, verbose : bool) -> None:
        self.__llm = llm
        self.__prompt = prompt
        self.__verbose = verbose
    
    def getConversationChain():
        conversation_chain = ConversationChain(
            prompt=self.__prompt,
            llm=self.__llm,
            verbose=self.__verbose,
            memory=ConversationSummaryBufferMemory(
                llm=self.__llm,
                max_token_limit=650
            )
        )
        # conversation.predict(input="What's the weather?")
        return conversation_chain

# For testing 
if __name__ == "__main__":
    
    from langchain_community.llms import LlamaCpp
    from prompt_data import CustomPromptData
    from langchain_community.vectorstores import FAISS
    from model import LLMModel
    from langchain_community.embeddings import HuggingFaceEmbeddings

    llmModel = LLMModel()
    llm = llmModel.getLLM()

    condense_prompt_data = """
        The current chat history is {chat_history}.

        User's current question is {question}
    """
    condense_prompt = CustomPromptTemplate(CustomPromptData(condense_prompt_data).get_prompt_data(), 
            input_variables=["question", "chat_history"]).getCustomPromptTemplate()

    combine_custom_data = """You are an AI assistant who helps the user with their questions. Given the context below, answer the user's question. If you don't know the answer, politely tell that you are unware of that and do not try to make an answer of your own.

        Context: {context}
        Question: {question}

        Return the answer below.
        Answer:
        """
    combine_custom_prompt = CustomPromptTemplate(CustomPromptData(combine_custom_data).get_prompt_data(), 
            input_variables=["context", "question"]).getCustomPromptTemplate()


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local("faiss_vector_store", embeddings)

    convRetrievalQAChainObject = ConversationalRetrievalQAChain(llm=llm, condense_prompt=condense_prompt, 
                                       combine_docs_custom_prompt= combine_custom_prompt, db=db)
    chain, memory = convRetrievalQAChainObject.getConversationalRetrievalQAChain()

    print(chain.invoke("When was pytorch developed?"))
    print(memory.buffer)
    print(chain.invoke("What is the latest version of PyTorch?"))
    print(memory.buffer)
