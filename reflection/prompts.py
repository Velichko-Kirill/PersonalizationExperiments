from langchain_core.prompts import PromptTemplate


biographies_reflection_prompt: PromptTemplate = PromptTemplate(
    template="""You are world class writer-biographer. Moreover, you are a literary professor. 
        Given the part of some biography, you should estimate it properly and provide feedback for another writer. 
        Especially take into account the psychological description of the character which biography is writing. 
        Provide detailed recommendations for literary expansion and enrichment the part you're given to.
        Part of biography: '''{chunk}'''
        psychological_description: '''{description}
        """,
    input_variables=["chunk", "description"]
)

event_reflection_prompt: PromptTemplate = PromptTemplate(
    template="""
    
    """,
    input_variables=[]
)

action_reflection_prompt: PromptTemplate = PromptTemplate(
    template="""

""",
    input_variables=[]
)

planning_reflection_prompt: PromptTemplate = PromptTemplate(
    template="""
    
    """,
    input_variables=[]
)

pass_psycho_test_prompt: PromptTemplate = PromptTemplate(
    template="""

""",
    input_variables=[]
)
