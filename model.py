from langchain_community.llms import LlamaCpp
from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

class LLMModel:
    def __init__(self, 
                model_name="Mistral-7b-v0.2-instruct", 
                model_path="../LLMs/mistral-7b-instruct-v0.2.Q6_K.gguf",
                n_gpu_layers=15,
                n_batch=256,
                temperature=0.1,
                verbose=True,
                max_tokens=512,
                top_p=1,
                use_cache=False):

        self.__model_name = model_name
        self.__model_path = model_path
        self.__n_gpu_layers = n_gpu_layers
        self.__n_batch = n_batch
        self.__temperature = temperature
        self.__verbose = verbose
        self.__max_tokens = max_tokens
        self.__top_p = top_p
        self.__use_cache = use_cache

        try:
            # Note - For some reason, by loading model using AutoModelForCausalLM, facing some not Runnable error
            
            # llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="../LLMs//llama-2-7b-chat.Q5_K_M.gguf", 
            #     model_type="llama", 
            #     gpu_layers=20,
            #     max_new_tokens=512,
            #     temperature=0.1)
            
            # llm = CTransformers(model = "../LLMs/llama-2-7b-chat.ggmlv3.q6_K.bin",
            #     model_type="llama", 
            #     max_new_tokens=512, 
            #     temperature=0.5
            # )

            self.__llm = LlamaCpp(
                model_path=self.__model_path,
                temperature=self.__temperature,
                max_tokens=self.__max_tokens,
                top_p=self.__top_p,
                n_gpu_layers=self.__n_gpu_layers,
                n_batch=self.__n_batch,
                verbose=self.__verbose,
            )
            print(f"Model {self.__model_name} loaded successfully!")

            if self.__use_cache:
                set_llm_cache(InMemoryCache())

            
        except Exception as e:
            print("Couldn't load the model. Exception : ", e)
    
    def getLLM(self):
        return self.__llm