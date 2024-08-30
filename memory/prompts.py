from langchain_core.prompts import PromptTemplate

retrieval_prompt: PromptTemplate = PromptTemplate(
    template="""
   Taking into account input query, find the most salient and relevant memory from 'context' to answer it.
   context: '''{context}'''
   
   input query: '''{input}'''
   answer: 
   """,
    input_variables=["context", "input"]
)

qa_prompt: PromptTemplate = PromptTemplate(
    template="""
    Taking into account the context provided, give me a precise and complete answer to the question.
    context: '''{context}'''
    question: '''{question}'''
    answer: 
    """,
    input_variables=["context", "question"]

)
