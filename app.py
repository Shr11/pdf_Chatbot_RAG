import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def get_response(uploaded_docs, api_key, query):
    
    if uploaded_docs is not None:
        # empty list
        docs = [uploaded_docs.read().decode()]
        
        # # read & decode pipeline
        # for up_file in uploaded_docs:
        #     docs.append(up_file.read().decode())
            
        # split , embed and process
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)  
        
        texts = text_splitter.create_documents(docs)
        
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # creating vectorstore
        db = Chroma.from_documents(texts, embeddings)
        
        # retriever interface
        retriever = db.as_retriever()
        
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
        
        return qa.run(query)
    
    
    
    
# Front Page
st.set_page_config(page_title="PDF answerer")
st.title('🔗 Ask the Document App')

st.header('About the App')
st.write("""
Upload text documents and receive answers to queries based on the content of these documents. Utilizing RAG approach powered by OpenAI's GPT models, the app provides insightful and contextually relevant answers.

### How It Works
- Upload a Document: You can upload any text document in `.pdf` format.
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.


### Get Started
Simply upload your document and start asking questions!
""")

uploaded_file = st.file_uploader("Upload PDF", type='pdf')

query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

result = []
with st.form('AskForm', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = get_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key



if len(result):
    st.info(response)
    
    