__all__ = [
    "high_level_bio_prompt",
    "biography_dividing_prompt",
    "examples",
    "expand_chunk_prompt",
    "summarize_prompt",

    "check_question",
]

from typing import List, Dict

summarize_prompt: str = """
    You are a world class writer. 
    You will provided py the part of some biography. 
    You should to summarize it in {num_words} words at maximum, keeping all essential facts. 
    Write a concise, well-structured summary keeping all the essential information from the source text.
    There is a biography part to be summarized: '{text}'
    In your answer leave the summarized text only, without any your comments such as 'Sure, I'd like to help...' or other information.
"""

high_level_bio_prompt: str = """
    You are a world class writer specializing on biographies.
    You will provided with the set of traits that some person would have. 
    You should provide me his (or her) hypothetical short biography that must lead to personal traits provided.
    The story you will write should contains all essential facts about his/her life such as education, career, 
    marriage, social relationships, awards, etc.
    
    There is a traits to obtain:
        '{traits}'
    
    In your answer leave the biography info only, without any your comments such as 'Sure, I'd like to help...' 
    or any other information.    
 """

psycho_desc_prompt: str = """
    You are a world class psychologist. 
    You will provided by the dictionary where values are depicting some person's relation to the entities which are a keys. 
    For example, the pair "friends" : "has many friends" reflects the tendency to establish a many friendship's connections.
    Carefully taking into account all those tendencies and using all your available psychology knowledge, provide as a result
    a brief summary of the psychology of that person. 
    Be insightful but laconic as a real psychologist or psychoanalytic. Try to discover the underlying system for a given set of traits.
    Your answer should be limits of one paragraph of 6-10 sentences, but completely cover everything needed for a person's behavior predicting.
    In your answer leave only analysis, without any introductory words.
    There is a dictionary:
     '''{psycho_profile}'''
    Your answer:
"""

expand_chunk_prompt: str = """
    You are a world class writer specializing on biographies.
    You will provided with the part of biography that represents some period of personal life. 
    You should enrich it with the details, expanding that part into a full story. 
    In doing so you must not go beyond the time limits described there.
    New information that you introduce, must not contradict any of the previous facts and events.
    The actions of main character should lie in the flow of his character.
    At least a twofold increasing the source text length is expected.
    
    There is a part of biography you have to enrich: 
        '{chunk}'
        
    Take into account the following numeric parameters (1 to 100 scale) 
        redundancy = {redundancy}
        importance = {importance}
    
    Its characterize the provided biography part. You should keep the balance, providing more details about the most
    important parts and being more laconic if the redundancy score is already high.
    
    Also take into account a psychological profile of the person whom biography is provided.
    psychological profile: '{psychological}'
        
    Return only text of the history, do not write any additional information such as "I'm glad to help..." etc.
"""

biography_dividing_prompt: str = """
    You will provided with the part of some person's biography. 
    You should highlight the essential facts of this life-story.
    Then, represent your answer as a list of those with corresponding dates. 
        
        There is a text you have to parse:
        ```{text}```
"""

examples: List[Dict[str, str]] = [
    {
        "text": """Juliya Skvorodina was born on December 21, 1988, in Russia. She is a highly intelligent and ambitious individual with a passion for writing. Juliya's birth in Russia exposed her to a diverse cultural background, which has greatly influenced her writing style and perspective. She has a unique ability to craft captivating stories that are both relatable and thought-provoking.
Throughout her life, Juliya has been driven by a desire to explore the world and experience new things. This curiosity has led her to travel extensively, both within Russia and abroad, and has broadened her horizons and deepened her understanding of the world.
Juliya's love of writing began at a young age, and she has spent countless hours honing her craft and developing her unique voice. Her dedication to her art is evident in every piece she creates, and her passion for writing is evident to all who read her work.""",

        "list of facts": """
            21.12.1988: Juliya Skvorodina's birth
            2003: First essai writing
            05.2010: First publication
            01.01.2012: Second publication
            ...      
        """}
]

next_action_template: str = """
     Below I provide you a part of biography formalized as \n
    {format_instructions}\n
    In your answer, provide one new hypothetical biography action in the same format.
        - It must not contradict other facts of life.
        - It should fit harmoniously into the overall structure of the biography.
        - The features changed should be logically related to the precedent actions.
        - The proposed event must date strictly back to a later date than the facts cited.  
        - It should not be one of the fact cited.
"""

check_question: str = """
    What the value of parameter `{field}` has this person?   
"""

answer_prompt: str = """
    Now you are passing a psychological test, aims to profile you.
    Considering all the provided information of your psychological traits from the biography of your role, estimate the 
    relevance of the utterance to you.  
    The utterance is: '''{utterance}'''
    If this utterance is irrelevant to your character: return -1
    If this utterance contradict to your character: return 1
    
    Return only one number as an answer without any additional symbols.  
"""
