## About

I know there are already a 100 repositories with examples or even entire application built using LLM. The aim of this repository to serve as a reference code base. We have Documentations, but they explain you only a certain function or block, but doesn't tell you how to plug it in actually. This repository if for all the Devs who are just starting their journey in the area of LLMs.

I will keep updating the repository, keep adding new contents as and when I will be coming across new things. Moving forward, as the content will increase, the entire repo will be restructed. Currently the repository is built on an exmaple RAG application.

## Medium Article

To complement the content in this repository, I have also prepared a detailed Medium article that covers the theoretical aspects of Large Language Models.

## Repository Contents

### ingest.py

The `ingest.py` file is a comprehensive tutorial that demonstrates how to use FAISS and Qdrant vector stores and databases. It walks you through each step, from loading data to splitting it, loading embedding models, and finally generating and storing embeddings.

### model.py

In `model.py`, you'll find instructions on how to load a model using the llama.cpp and CTransformers library. 

### chains.py

The `chains.py` file showcases how to create different types of chains, including Retrieval, ConversationalQARetrieval, and normal Conversational chains. These examples demonstrate how to leverage LLMs for various tasks and applications.

### prompt.py

`prompt.py` offers insights into creating a prompt template. This file demonstrates how to structure prompts to interact with LLMs effectively and obtain desired responses.

### Prompt_data

The `Prompt_data` contains a class that stores data for the prompt template. This resource is useful for organizing and managing prompt-related data efficiently.

### Test.py

Finally, `Test.py` integrates all the above modules and builds a sample RAG (Retriever-Augmented Generation) application for PDFs. This file ties together the concepts and techniques discussed in the repository, providing a practical demonstration of LLM usage.

## Getting Started

To get started with this repository, simply clone it to your local machine and explore the files and examples provided. Follow along with the tutorials and experiments to gain hands-on experience with Large Language Models.

If you already have your dev environment ready for LLMs then start from step 4 else follow all the steps.

## Setting Up Development Environment

### Step 1: Pull Docker Image
Pull the Docker image with CUDA and other libraries:
```bash
docker pull sahilchachra/pytorch_dev:v0.1
```

### Step 2: Clone the Repository
Clone the repository:
```bash
git clone git@github.com:SahilChachra/LLMReference.git
```

### Step 3: Create Docker Container
Create a Docker container and mount the code base to the container:
```bash
export DISPLAY=:1 && xhost +
sudo docker create -it --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v PATH_TO_CODE:/home/ --name DEV_CONTAINER sahilchachra/pytorch_dev:v0.1
```

### Step 4: Prepare Data
Create a folder named data in the cloned repository. Place the PDFs you want to run your RAG (Relevance Aggregation Generator) on, and update the same in ingest.py and run following. 
```bash
python3 ingest.py
```

### Step 5: Execute into the Container
Execute into the container:
```bash
sudo docker exec -it DEV_CONTAINER bash
```

### Step 6: Run the Test Script
Navigate to the repository directory and run test.py:
```bash
cd /home/LLMReference
python3 test.py
```
