class CustomPromptData:
    def __init__(seld):
        self.__custom_prompt_data = """You are an AI assistant who helps the user with their questions. Given the context below, answer the user's question. If you don't know the answer, politely tell that you are unware of that and do not try to make an answer of your own.

        Context: {context}
        Question: {question}

        Return the answer below.
        Answer:
        """
    
    def get_prompt_data(self):
        return self.__custom_prompt_data