class LLM:
    def __init__(self, 
                model_name="Mistral-7b-v0.2-instruct", 
                model_path="../LLMs/mistral-7b-instruct-v0.2.Q6_K.gguf",
                n_gpu_layers=15,
                n_batch=256,
                temperature=0.1,
                verbose=true,
                max_tokens=512)

        self.model_name = model_path
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.temperature = temperature
        self.verbose = verbose
        self.max_tokens = max_tokens

        try:'
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

            self.llm = LlamaCpp(
                model_path=self.model_path,
                temperature=self.temperature, # 0.1 : to the point answer and least random, 0.9 : make stories and be more random
                max_tokens=self.max_tokens, # Maximum number of token to output
                top_p=1,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch, # Should be less than max_token as per docs
                verbose=self.verbose,  # Verbose is required to pass to the callback manager
            )
            print(f"Model {self.model_name} loaded successfully!")
            
        except Exception as e:
            print("Couldn't load the model. Exception : ", e)
    
    def getLLM():
        return self.llm