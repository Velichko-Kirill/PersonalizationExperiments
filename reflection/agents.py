from typing import Annotated, Union, Any, Dict
from typing_extensions import TypedDict

from langchain_core.prompts import PromptTemplate
from langchain_experimental.generative_agents import GenerativeAgent
from langchain_ollama import ChatOllama
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import MessageGraph

"""
reflection_prompt choice depends on the task:
    - Init --> biographies_reflection_prompt
    - Memory --> event_reflection_prompt
    - Planning --> planning_reflection_prompt
    - Action --> action_reflection_prompt
    - Evaluation  --> pass_psycho_test_prompt
"""


class ReflectionAgent(MessageGraph): ...

