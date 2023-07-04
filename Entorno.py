from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import glob

def peticion_gpt(word_1,word_2):
    '''   # Cargar nuevos datos a Pinecone # 
    ### Extraer papers 
    file_list = glob.glob('papers2/*.txt')
    documentos = []
    for file_path in file_list:
        loader = UnstructuredFileLoader(file_path)
        data = loader.load()
        documentos.append(data)

    ### Dividir papers en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0)
    documentos_chunks = []
    for documento in documentos:
        chonks = text_splitter.split_documents(documento)
        documentos_chunks.extend(chonks)

    ### Conecto a OpenAI para embeddings y a Pinecone para Vectores
    embeddings = OpenAIEmbeddings(
        openai_api_key='OPENI_KEY')

    pinecone.init(
        api_key = '4f58d18b-75ff-4404-94bc-832bf24c45d1',
        environment = 'asia-southeast1-gcp-free')
    ## Subir datos
    docsearch = Pinecone.from_texts([t.page_content for t in documentos_chunks], embeddings, index_name=index_name)
    '''

    #Coneccion Pinecone
    pinecone.init(
        api_key = '4f58d18b-75ff-4404-94bc-832bf24c45d1',
        environment = 'asia-southeast1-gcp-free'
    )
    index_name = 'tid'
    #Coneccion OpenAI (LLM y Embeddings)
    llm = OpenAI(
        temperature = 0,
        openai_api_key = 'OPENAI_APIKEY',
        )
    index_name = 'tid1'
    embeddings = OpenAIEmbeddings(openai_api_key='OPENAI_APIKEY')

    docsearch1 = Pinecone.from_existing_index(index_name, embeddings)
    
    chain = load_qa_chain(llm, chain_type="stuff")
    query = f"Describa en formato de regla, 5 relaciones posible entre {word_1} y {word_2}"
    print(query)
    docs = docsearch1.similarity_search(query)
    print(chain.run(input_documents = docs, question = query))

    return docs

