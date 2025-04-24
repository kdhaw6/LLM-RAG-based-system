import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers
from transformers import AutoModel
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Load the PDF files from the path
loader = DirectoryLoader(r'C:\Users\Kunal\Downloads\Hugging Face RAG\data', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q4_0.bin', callbacks=[StreamingStdOutCallbackHandler()])


# Vector store
vector_store = None
try:
    vector_store = FAISS.from_documents(text_chunks, embeddings)
except IndexError as e:
    st.error(f"An error occurred during vector store creation: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

if vector_store is None:
    st.error("Failed to create vector store.")
    st.stop()

# Create LLM
#llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
#                    config={'max_new_tokens': 128, 'temperature': 0.01})



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

st.title("Your AI Tech Help")

def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I'm here to listen you🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about attention in AI", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()