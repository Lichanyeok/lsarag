#pip install -U langchain openai faiss-cpu tiktoken PyPDF2 beautifulsoup4 requests streamlit langchain-community pypdf python-dotenv google-api-python-client langchain-openai

# 청킹
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


# 임베딩
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
# from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate

# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
# retriever = faiss_vectorstore.as_retriever(k=3)

# prompt = PromptTemplate(
#   input_variables=["context","question"],
#   template="""
#     당신은 근로기준법을 전문적으로 분석하여 법적 질문에 답변을 하는 전문 법률가입니다.
#     아래 근로기준법 관련 문서에서 취득한 정보를 바탕으로 question에 적절한 답변을 부탁합니다.
#     ---
#     {context}
#     ---
#     단, 질문이 위의 근로기준법 내용과 관련이 없을 시 "해당 질문은 근로기준법 관련 내용이 아닙니다." 라는 답변으로 통일해줘.
#     근로기준법과 관련된 내용일 경우 법적 근거를 명확하게 제시한 답변을 간략하게 요약하여 전달해줘.
#     ---
#     question:{question} 
#   """
# )

# # rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
# chat_chain = ConversationalRetrievalChain.from_llm(
#   llm = llm,
#   retriever = retriever,
#   condense_question_prompt=prompt,
#   verbose=True
# )

# chat_chain({"question":"쿠버네티스 사용법을 알려줘","chat_history":[]})
# '''
# 예상 답변 
# {'question': '쿠버네티스 사용법을 알려줘',
#  'chat_history': [],
#  'answer': '해당 질문은 근로기준법 관련 내용이 아닙니다. 
#  실제 답변
#  {'question': '쿠버네티스 사용법을 알려줘',
#  'chat_history': [],
#  'answer': '죄송하지만, 쿠버네티스의 사용법에 대한 구체적인 정보를 제공할 수 없습니다. 쿠버네티스는 컨테이너화된 애플리케이션의 배포, 확장 및 관리를 자동화하는 오픈 소스 플랫폼입니다. 사용법에 대한 자세한 정보는 공식 문서나 관련 튜토리얼을 참고하시기 바랍니다. 공식 문서는 [Kubernetes 공식 웹사이트](https://kubernetes.io/docs/)에서 확인할 수 있습니다.'

#  왜 예상한대로 프롬프트가 작동하지 않았을까 ? 
#  - ConversationalRetrievalChain체인의 작동 구조를 알아야함. condense_question_prompt에 매핑한 프롬프트는 완성형 답변을 주는것에 적용 하는 프롬프트가 아니라.
#   채팅의 불완전한 질문을 보완할 때 사용되기 때문임. 따라서 해당 프롬프트는 context(임베딩해서 들고온 문서)를 참조하지 못하게 됨 -> 답변을 완성시킬 때 사용되는 프롬프트가 아니기 때문.
#   따라서 체인을 결합하는 식으로 구성해야 의도한 대로 작동하게 됨
# '''


# 프롬프트 내부구조를 고려한 체인 설계
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain,ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
retriever = faiss_vectorstore.as_retriever(k=3)

#불완전한 질문을 재구성하는 프롬프트
question_prompt = PromptTemplate(
  input_variables=["question","chat_history"],
  template="""
  다음 대화를 참고하여 명확한 질문으로 재작성하세요:
  대화 : {chat_history}
  질문 : {question}
"""
)

#generator 생성
question_generator = LLMChain(llm=llm, prompt=question_prompt)

#원하는 답변의 최종 템플릿
custom_answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    당신은 근로기준법을 전문적으로 분석하여 법적 질문에 답변을 하는 전문 법률가입니다.
    아래 근로기준법 관련 문서에서 취득한 정보를 바탕으로 question에 적절한 답변을 부탁합니다.
    ---
    {context}
    ---
    단, 넌 근로기준법과 관련된 내용에만 답변을 할 의무가 있어.
    질문이 근로기준법 내용과 완전히 관련 없을 시 "해당 질문은 근로기준법 관련 내용이 아닙니다." 라는 답변으로 통일해줘. 예를 들면, "음식의 레시피를 질문하거나, 유행하는 노래를 묻는다거나, 특정 인물에 대해 질문 하는 것 처럼 말이야.
    하지만, 위의 근로기준법 내용과 조금 일치하지 않더라도 노동 혹은 근무에 관한 내용은 찾아서 답변해줘
    모든 답변은 법적 근거를 명확하게 제시하고 간략하게 요약하여 전달해줘.
    ---
    질문: {question}
    """)
    #context와 관련이 없더라도 근로기준에 관하여 답변 유도 되도록 프롬프트 수정
    #question : 최저임금 기준 주 60시간 근무 시 주휴수당은 얼마야 ?
    #asis -> 해당 질문은 근로기준법 관련 내용이 아닙니다.
    #tobe -> 
    # 주휴수당은 근로자가 1주 동안 소정의 근로일을 개근한 경우에 지급되는 유급휴일에 대한 수당입니다. 주휴수당은 1주일 동안의 소정근로시간에 대한 임금을 기준으로 산정됩니다.
    # 주휴수당 계산 방법:

    # 주휴수당 = (1주 소정근로시간 / 40시간) × 8시간 × 시급
    # 주 60시간 근무 시:

    # 주 60시간 근무는 연장근로가 포함된 시간입니다. 주휴수당은 소정근로시간(기본 근로시간)에 대해서만 계산됩니다.
    # 소정근로시간이 주 40시간이라면, 주휴수당은 다음과 같이 계산됩니다:
    # 주휴수당 = (40시간 / 40시간) × 8시간 × 최저시급
    # 최저임금 기준:

    # 2023년 기준 최저시급이 9,620원이라면:
    # 주휴수당 = 8시간 × 9,620원 = 76,960원
    



qa_llm_chain = LLMChain(llm=llm, prompt=custom_answer_prompt)
combine_docs_chain = StuffDocumentsChain(
    llm_chain=qa_llm_chain,
    document_variable_name="context"
)


#최종 체인 구성
chat_chain = ConversationalRetrievalChain(
  retriever = retriever,
  question_generator = question_generator,
  combine_docs_chain = combine_docs_chain,
  verbose = True
)

# google 검색 체인
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



# 스트림릿 띄워서 체인 run
import streamlit as st

st.title("근로기준법 챗봇")
query = st.text_input("궁금한 점을 입력하세요:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    with st.spinner("답변 생성 중..."):
        result = chat_chain({"question": query, "chat_history":st.session_state.chat_history})
        st.session_state.chat_history.append((query, result["answer"]))
        
        st.subheader("📘 법령 기반 답변")
        st.write(result["answer"])

        st.subheader("💬 이전 대화 기록")
        for user_q, bot_a in st.session_state.chat_history:
            st.markdown(f"사용자 🚀 : {user_q}")
            st.markdown(f"답변 :sunglasses: : {bot_a}")
        # 외부 사례 검색 및 요약
        search_results = search.results(query=query + " 판례", num_results=1)
        if not search_results[0]['Result']:
            web_content = search_results[0]['snippet']
            summary = summary_chain.run(content=web_content)
            st.subheader("📌 실제 사례 요약")
            st.write(summary)
            st.markdown(f"🔗 [출처 링크]({search_results[0]['link']})")

        


