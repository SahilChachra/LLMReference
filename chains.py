from langchain.chains import RetrievalQA
from model import LLMModel
from prompt_template import CustomPromptTemplate
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory


class RetrievalQAChain:
    """
        Use when working with RAGs
    """
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
    """
        Use when you want your RAG application to have memory
    """
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
    """
        Use it when you want to build a simple conversational chain
    """
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
        # conversation.predict(input="What's the weather today?")
        return conversation_chain
