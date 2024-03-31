from flask import Flask, render_template, request
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
import os

# Load environment variables from .env file
load_dotenv()

# Get Pinecone API key and environment from environment variables
PINECONE_API_KEY = os.environ.get('93698fd9-7c0a-4cf6-ae61-500b34cb7851')
PINECONE_API_ENV = os.environ.get('gcp-starter')

# Initialize Flask app
app = Flask(__name__)

# Download Hugging Face embeddings - you need to implement this function
def download_hugging_face_embeddings():
    pass

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Define index name
index_name = "5G_OPTI_CHATBOT"

# Load the index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Define prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Define chain type kwargs
chain_type_kwargs = {"prompt": PROMPT}

# Initialize CTransformers
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens':512, 'temperature':0.8})

# Initialize RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
                                  return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

# Route for rendering chat.html template
@app.route("/")
def index():
    return render_template('chat.html')

# Route for receiving messages and returning responses
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(input_text)
    result = qa({"query": input_text})
    print("Response:", result["result"])
    return str(result["result"])

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
