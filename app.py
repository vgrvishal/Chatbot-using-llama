from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import markdown
from llama_index.core import VectorStoreIndex,  Document , Settings
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.core import SimpleDirectoryReader
import os
from llama_index.llms.groq import Groq
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

model_name = "mixedbread-ai/mxbai-embed-large-v1"
embed_model =  MixedbreadAIEmbedding(api_key='emb_b44a9094713b8597f7999596d2bdf518d2f6dcab96441951', model_name=model_name)

reader = SimpleDirectoryReader(input_files= ['D:/medical/data.pdf'])
docs = reader.load_data()
  
llm = Groq(model="llama-3.1-8b-instant", api_key="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")
Settings.llm = llm
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(docs)
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# print(chat_engine.chat("what is medicine"))
def format_response(text):
    """
    Convert the response text into HTML format using markdown.
    """
    formatted_text = f"{text}"
    html = markdown.markdown(formatted_text)
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
        # Get the answer from the LangChain model
        response =chat_engine.chat(user_input)
        formatted_response = format_response(response)  # Format the answer into HTML
        return jsonify({"response": formatted_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode for better error tracking