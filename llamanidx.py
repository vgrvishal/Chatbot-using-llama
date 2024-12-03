from flask import Flask, jsonify, request
from llama_index.core import VectorStoreIndex, ServiceContext, Document, Settings
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
import markdown
import os

# Initialize the Flask app
app = Flask(__name__)

# Global variable for chat engine
chat_engine = None

def init_models():
    """
    Initialize the LLM and embedding model settings.
    """
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    embed_model = MixedbreadAIEmbedding(
        'emb_b44a9094713b8597f7999596d2bdf518d2f6dcab96441951',
        model_name=model_name
    )
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key="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"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

def initialize_chat_engine():
    """
    Initialize the chat engine from document data.
    """
    global chat_engine
    init_models()
    reader = SimpleDirectoryReader(input_files=['D:/medical/data.pdf'])
    docs = reader.load_data()

    index = VectorStoreIndex.from_documents(docs)
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

def format_response(text):
    """
    Convert the response text into HTML format using markdown.
    """
    html = markdown.markdown(text)
    return html

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles POST requests to the /chat endpoint.
    Expects a JSON body with a 'message' field.
    """
    user_input = request.json.get('message')
    print(user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Get the answer from the chat engine
        response = chat_engine.chat(user_input)
        formatted_response = format_response(response)
        return jsonify({"response": formatted_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the chat engine before running the app
    initialize_chat_engine()
    app.run(debug=True)
