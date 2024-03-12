from langchain.prompts import PromptTemplate

class CustomPromptTemplate:
    def __init__(self, prompt_data, input_variables):
        self.__prompt = PromptTemplate(template=prompt_data,
                            input_variables=input_variables)
    
    def getCustomPromptTemplate():
        return self.__prompt