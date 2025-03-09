This folder is to be used to store the code for your project. 

Screenshots provided for reference.
- CLEARML: clearml.png
- MongoDBCluster: mongodb.png
- QDRANT: qdrant.png
- OLLAMA: ollama.png
- app: app-question1.png app-question2.png app-question3.png


## How to run:
 - Local Environment Setup
    - docker file is provided with screenshot however we are using online mongo and qdrant for better collaboration and scalability if required(hypothetical scenario/ benefit) ...this is not required to run the rest of the code.
 
 - Run the colab notebook(ETL_FEATURIZING_RAG.ipynb) to do the following: 
    - Connect to an online MongoDB database
    - Setup clearml orchestration
    - download ros2 documentation
    - parse it chunk it and store them in mongodb 
    
    - generate its embeddings and store them as well in mongodb database (backup)

    - Connect to the qdrant cloud cluster 
    - store the embeddings in its engine
    - sample retrieval is showcased.
    
 - Launch Ollama server using the below code:
    - curl -fsSL https://ollama.com/install.sh | sh # for linux
    - ollama run hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

- Instruction Dataset and Finetuning:
   - instruction_dataset.csv
   - finetune.ipynb (has code to generate the instruction_dataset and finetune the model)
     
 - Launch APP:
    - to setup (pip install gradio transformers sentence-transformers qdrant-client requests)  
    - to run the app use "python app4.py"
    - It performs retrieval using qdrant 
    - It has a function for prompting rag and getting the response from the ollama server
    - It has dropdown of questions 
    - It has a chatbot style interface (please wait for the Q/A to be shown together)
  You can either select from the dropdown (select and hit submit and wait) or type your own question (type hit submit and wait)

~
