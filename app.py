from flask import Flask, jsonify, request
import requests
from bs4 import BeautifulSoup
from flask_cors import CORS
# from flask_ngrok import run_with_ngrok
import pickle
import faiss
from langchain.vectorstores import FAISS
import os
# InstructorEmbedding
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# from pyngrok import ngrok

# Set your ngrok authtoken here
# ngrok.set_auth_token("2UdEoJp3eYipl5jphZzH0bwS3lJ_5RPAeCVfBmfCTjxVKHgBk")



app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return "Hello, world!"

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})
db_instructEmbedd = FAISS.load_local("faiss_index", embeddings)
@app.route("/qa", methods=["POST"])
def qa():
    query = request.json["query"]

    # Load the embeddings
    # embeddings_path = os.path.join(os.getcwd(), "index.pkl")
    # with open(embeddings_path, "rb") as f:
    #     embeddings = pickle.load(f)

    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
    #                                                   model_kwargs={"device": "cuda"})
    # db_instructEmbedd = FAISS.load_local("faiss_index", embeddings)

    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    retriever.search_type
    retriever.search_kwargs
    docs = retriever.get_relevant_documents("tell me am I infertility. How can I check it ")
    
    
    
    print(docs[0],docs[1],docs[2])
    print(len(docs[0].page_content),len(docs[1].page_content),len(docs[2].page_content))
    
    
    # create the chain to answer questions 
    import os
    from langchain.chains import RetrievalQA
    
    os.environ["OPENAI_API_KEY"] = process.env.OPENAI_API_KEY
    qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0, ), 
          chain_type="stuff", 
          retriever=retriever, 
          return_source_documents=True)




## Cite sources

    import textwrap
    
    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')
    
        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text
    
    def process_llm_response(llm_response):
        text=wrap_text_preserve_newlines(llm_response['result'])
        print(text)
        print('\nSources:')
        sources=[]
        for source in llm_response["source_documents"]:
            sources.append(source.metadata['source'])
            print(sources)
        result =text + '\nSources:' + '\n'.join(sources)
        # texts = result.split('\n')
        # {% for para in texts %}
        #     <p>{{para}}</p>
        # {% endfor %}
        # text = texts.replace('\n', '<br>')
        # {% autoescape false %}
        #     {{text}}
        # {% endautoescape %}

        print("result",result)    
        return text 


    query = query
    print('-------------------Instructor Embeddings------------------\n')
    llm_response = qa_chain_instrucEmbed(query)
    result=process_llm_response(llm_response)
    print("result",result)
    return jsonify(result)


if __name__ == "__main__":
    run_with_ngrok(app)
    app.run()
