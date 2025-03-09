import subprocess
import time
import requests
import json
import gradio as gr
from qdrant_client import QdrantClient
# Model path
model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
context = []


# Connect to your Qdrant Cloud instance
qdrant_client = QdrantClient(
    url="https://f5cfd4dd-6224-4316-8efb-a7957d2ad826.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="6_rN_0sytviivkKIMxgErXOEeGlvgW46ZV5Pehnu4k7B525TAEGUUg"
)



def ensure_ollama_running(model_path):
    """
    Ensures that the Ollama server is running and attempts to start it with the specified model.
    """
    print("Checking if Ollama is running...")
    try:
        # Test if Ollama server is responding
        response = requests.get("http://localhost:11434", timeout=3)
        if response.status_code == 200:
            print("Ollama is running.")
    except requests.ConnectionError:
        print("Ollama is not running. Attempting to start it...")
        process = subprocess.Popen(
            ["ollama", "run", model_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(10)  # Allow time for the server to start
        stdout, stderr = process.communicate()
        if stdout:
            print(f"Ollama output:\n{stdout}")
        if stderr:
            print(f"Ollama error details:\n{stderr}")
        raise RuntimeError("Failed to start Ollama. Check logs for details.")

# Ensure Ollama is running
ensure_ollama_running(model)

def generate(prompt, context):
    """
    Sends a request to the Ollama `generate` endpoint and returns the generated response.
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'context': context,
            },
            stream=False
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to communicate with Ollama: {e}")

    response_text = ""
    for line in response.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        print(response_part)
        if 'error' in body:
            raise Exception(body['error'])
        response_text += response_part

        if body.get('done', False):
            context = body.get('context', [])
            return response_text, context




# 

def search_in_qdrant_cloud(query, qdrant_client, qdrant_collection_name, embedding_model_name="all-MiniLM-L6-v2", top_k=1):
    """
    Performs vector search in Qdrant Cloud given a query and returns the top K results.
    """
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    model = SentenceTransformer(embedding_model_name)

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=qdrant_collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    # Process and return results
    results = [
        {
            "score": result.score,
            "file_name": result.payload["file_name"],
            "file_path": result.payload["file_path"],
            "content": result.payload["content"]
        }
        for result in search_results
    ]

    print("Qdrant - ok ")
    return results




def answer_question1(input_question, chat_history):
    """
    Retrieves context using Qdrant, generates a response using Ollama, 
    and updates chat history while preserving conversational context.
    """
    global context  # To maintain and update the conversation context
    chat_history = chat_history or []  # Initialize chat_history if it is None

    # Qdrant Collection Details
    qdrant_collection_name = "vectors"

    # Retrieve context from Qdrant
    results = search_in_qdrant_cloud(input_question, qdrant_client, qdrant_collection_name)
    print(f"Qdrant results: {results}")

    # Combine retrieved context (if available)
    if results and 'content' in results[0]:
        retrieved_context = "\n".join(result['content'] for result in results[:3])
    else:
        retrieved_context = "No relevant context found."

    print(f"Retrieved context for generation: {retrieved_context}")

    # Combine chat history context with retrieved context
    full_context = f"{context}\n{retrieved_context}" if context else retrieved_context

    # Generate a response
    try:
        output, new_context = generate(input_question, full_context)
        context = new_context  # Update the global context for continuity
    except Exception as e:
        output = f"Error during generation: {e}"

    # Update chat history
    chat_history.append((input_question, output))
    return chat_history, chat_history


def generate1(prompt, context):
    """
    Sends a request to the Ollama `generate` endpoint and returns the generated response.
    """
    try:
        print(f"Sending prompt: {prompt}")
        print(f"Using combined context: {context}")

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'context': context or "",  # Ensure context is always a string
            },
            stream=False
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to communicate with Ollama: {e}")

    response_text = ""
    for line in response.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        print(response_part)
        if 'error' in body:
            raise Exception(body['error'])
        response_text += response_part

        if body.get('done', False):
            context = body.get('context', [])  # Update context from Ollama response
            return response_text, context





def answer_question(input_question, chat_history):
    """
    Processes the input question (dropdown or custom) and generates an answer using Ollama.
    """
    global context
    chat_history = chat_history or []  # Initialize chat_history if it is None

    qdrant_collection_name="vectors"
    collection_name="parsed_files"
    
    # query = "Explain the concept of ROS nodes"
    results = search_in_qdrant_cloud(input_question, qdrant_client, qdrant_collection_name)
    print(results)


    prompt = f"### Question: {input_question} ###Context: {results[0]['content']}"
    
    output, context = generate(input_question, context)
    chat_history.append((input_question, output))
    return chat_history, chat_history

# Pre-defined questions for the dropdown menu
questions = [
    "Tell me how can I navigate to a specific pose - include replanning aspects in your answer.",
    "Can you provide me with code for this task?",
    "Can you explain ROS nodes"
]

# Gradio interface
with gr.Blocks() as block:
    gr.Markdown("<h1><center>RAG System Interaction</center></h1>")
    
    with gr.Row():
        question_dropdown = gr.Dropdown(
            choices=questions,
            label="Select a Question (or type below)",
            value=questions[0],
            elem_id="dropdown"
        )
    
    with gr.Row():
        custom_question = gr.Textbox(
            label="Or, type your custom question here:",
            placeholder="Type your question...",
            elem_id="custom_input"
        )
    
    chatbot = gr.Chatbot()
    state = gr.State()
    
    with gr.Row():
        submit = gr.Button("Submit", elem_id="submit_button")
    
    # Combine dropdown and custom input to allow either option
    def process_input(dropdown_question, typed_question):
        return typed_question if typed_question.strip() else dropdown_question

    submit.click(
        fn=lambda dropdown_question, typed_question, chat_history: answer_question(
            process_input(dropdown_question, typed_question), chat_history
        ),
        inputs=[question_dropdown, custom_question, state],
        outputs=[chatbot, state]
    )

# Add custom styles for noticeable input and submit button
block.css = """
#dropdown {
    background-color: #f0f8ff;
    border: 2px solid #6495ed;
    color: #000;
    font-weight: bold;
}

#custom_input {
    background-color: #ffffe0;
    border: 2px solid #ffd700;
    color: #000;
    font-weight: bold;
}

#submit_button {
    background-color: #32cd32;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px;
}
"""

block.launch(debug=True)
