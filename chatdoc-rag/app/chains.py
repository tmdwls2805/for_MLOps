from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

def build_qa_chain(llm_pipeline, retriever):
    prompt_template = """다음 문서를 참고해서 질문에 정확히 답해주세요.
문서 내용에서 답을 찾을 수 없으면 "정보를 찾을 수 없습니다"라고 답하세요.

문서: {context}

질문: {question}

답변:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
