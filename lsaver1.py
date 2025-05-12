
# %%
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./근로기준법.pdf")
pages = loader.load() #pages와 pdf의 총 장수는 같음

cleaned_pages = [
    page.__class__(page.page_content.replace('\n', ' ').strip(), metadata=page.metadata)
    for page in pages
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size =2000, chunk_overlap=500,)
document = text_splitter.split_documents(cleaned_pages)


# %%
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
import os

load_dotenv()

open_api_key = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_documents(document,embeddings)
faiss_vectorstore.save_local("./lsa_vectorstore")

# %%
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
retriever = faiss_vectorstore.as_retriever(k=3)
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


# %%
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv

load_dotenv()
search = GoogleSearchAPIWrapper()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

summary_prompt = PromptTemplate(
  input_variables=["content"],
  template=""" 다음 웹페이지 내용의 핵심을 간단히 요약하고, 법적 사례인지, 뉴스인지 구분해서 요약해줘:
    {content}
  """
)

summary_chain = LLMChain(llm=llm,prompt=summary_prompt)



# %%
import streamlit as st

st.title("근로기준법 챗봇")
query = st.text_input("궁금한 점을 입력하세요:")

if query:
    with st.spinner("답변 생성 중..."):
        rag_result = rag_chain(query)
        st.subheader("📘 법령 기반 답변")
        st.write(rag_result["result"])

        # 외부 사례 검색 및 요약
        search_results = search.results(query=query, num_results=1)
        try: 
            web_content = search_results[0]['snippet']
            summary = summary_chain.run(content=web_content)
            st.subheader("📌 실제 사례 요약")
            st.write(summary)
            st.markdown(f"🔗 [출처 링크]({search_results[0]['link']})")
        except:
            st.subheader("📌 조회된 사례가 없습니다")

