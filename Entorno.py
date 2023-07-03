from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import glob


''' ### Cargar datos a pinecone

### Extraer papers 
file_list = glob.glob('papers/*.txt')
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

### Subir datos
#docsearch = Pinecone.from_texts([t.page_content for t in documentos_chunks], embeddings, index_name=index_name)
'''


### Conectarse con los datos ya existentes en Pinecone
pinecone.init(
    api_key = '4f58d18b-75ff-4404-94bc-832bf24c45d1',
    environment = 'asia-southeast1-gcp-free'
)
index_name = 'tid'
embeddings = OpenAIEmbeddings(openai_api_key='OPENAI_APIKEY')
docsearch1 = Pinecone.from_existing_index(index_name, embeddings)

### OpenAI LLM + LangChain para el QA
llm = OpenAI(
    temperature = 0,
    openai_api_key = 'OPENAI_APIKEY',
    )
chain = load_qa_chain(llm, chain_type="stuff")

query = "Quienes son los autores?"
docs = docsearch1.similarity_search(query)
print(chain.run(input_documents = docs, question = query))
