from langchain.chains import RetrievalQA
from model import LLMModel
from prompt_template import CustomPromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory


class RetrievalQAChain:
    def __init__(self, llm : LLMModel, prompt : CustomPromptTemplate, db : any, 
                chain_type : str, return_source_documents : bool):
        self.__llm = llm
        self.__prompt = prompt
        self.__db = db
        self.__chain_type = chain_type
        self.__return_source_documents = return_source_documents

    def retrievalQAChain(self) -> RetrievalQA:
        qna_chain = RetrievalQA.from_chain_type(llm=self.__llm,
                                            chain_type=self.__chain_type,
                                            retriever=self.__db.as_retriever(search_kwargs={'k': 2}),
                                            return_source_documents=self.__return_source_documents,
                                            chain_type_kwargs={'prompt': self.__prompt}
                                        )
        return qna_chain


class ConversationalRetrievalQAChain:
    def __init__(self, llm : LLMModel, prompt : CustomPromptTemplate, db : any, 
                chain_type : str):
        self.__llm = llm
        self.__prompt = prompt
        self.__db = db
        self.__chain_type = chain_type
    
    def conversationalRetrievalQAChain(self) -> ConversationalRetrievalChain:
        conv_chain = ConversationalRetrievalChain.from_chain_type(
            chain_type=self.__chain_type,
            retriever=self.__db.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=self.__return_source_documents,
            chain_type_kwargs={'prompt': self.__prompt}
        )


class Conversation:
    def __init__(self, llm : LLMModel, prompt : CustomPromptTemplate, verbose : bool):
        self.__llm = llm
        self.__prompt = prompt
        self.__verbose = verbose
    
    def conversationChain():
        conversation_chain = ConversationChain(
            prompt=self.__prompt,
            llm=self.__llm,
            verbose=self.__verbose,
            memory=memory=ConversationSummaryBufferMemory(
                llm=self.__llm,
                max_token_limit=650
            )
        )
        # conversation.predict(input="What's the weather?")
        return conversation_chain
