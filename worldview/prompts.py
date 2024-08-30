from langchain_core.prompts import PromptTemplate

quest_1_prompt: PromptTemplate = PromptTemplate(
    template="""
    You are passing a psychological test aims to identify your personality type.
    While answering take into account your complete psychological description: 
    '{summary}'
    Answer in full accordance with this description.
    
    Given the utterance provided, choose to what extent do you agree with the statement:
    Utterance: {utterance}.
    
    Choose only one of these variants:
    - I completely agree
    - I rather agree
    - I somewhat agree
    - I neither agree nor disagree
    - I somewhat disagree
    - I rather disagree
    - I completely agree
    
    In your answer provide only the variant chosen, without any additional info.
    Your answer: 
    """,
    input_variables=["utterance", "summary"]
)

quest_2_prompt: PromptTemplate = PromptTemplate(
    template="""
        You are passing a psychological test aims to identify your personality type.
    While answering take into account your complete psychological description: 
    '{summary}'
    Answer in full accordance with this description.
    
    You are provided by the  utterance and two possible variants of it's continue:
    Utterance: {utterance}
    variant A: {answer_a}
    variant B: {answer_b}
    
    Choose one and only one of these variants. 
    In your answer provide only the letter of variant chosen, without any additional info.
    For example, if you have chosen variant 'A', return just 'A'
    Your answer:
    """,
    input_variables=["utterance", "summary", "answer_a", "answer_b"]
)


