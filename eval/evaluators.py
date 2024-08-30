import logging
import sys
from typing import Optional, Callable, Union

from langchain_ollama import OllamaLLM, ChatOllama
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.evaluation.integrations._langchain import SingleEvaluatorInput
from langsmith.schemas import Example, Run

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def get_correctness_evaluator(llm: Union[OllamaLLM, ChatOllama],
                              prepare_data: Optional[Callable[[Run, Optional[Example]],
                              SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    """Checks the global correctness of the answer, doesn't taking into account the context"""
    config = {
        "criteria": {
            "correctness": "Is the submission correct?"
        },
        "llm": llm
    }

    return LangChainStringEvaluator(
        "criteria",
        config=config,
        prepare_data=prepare_data
    )


def get_relevancy_evaluator(llm: Union[OllamaLLM, ChatOllama],
                            prepare_data: Optional[Callable[[Run, Optional[Example]],
                            SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    """Checks the relevancy of the answer, taking into account the context"""

    config = {
        "criteria": {
            "accuracy": "How accurate is this prediction compared to the reference on a scale of 1-10?"
        },
        "normalize_by": 10,
        "llm": llm
    }

    return LangChainStringEvaluator(
        "labeled_criteria",
        config=config,
        prepare_data=prepare_data
    )


def get_faithfullness_evaluator(llm: OllamaLLM,
                                prepare_data: Optional[Callable[[Run, Optional[Example]],
                                SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    """Checks the correctness of the answer, taking into account the context"""
    config = {
        "criteria": "",
        "llm": llm,
    }

    return LangChainStringEvaluator(
        "labeled_criteria",
        config=config,
        prepare_data=prepare_data

    )


def get_context_accuracy_evaluator(llm: OllamaLLM,
                                   prepare_data: Optional[Callable[[Run, Optional[Example]],
                                   SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    config = {
        "criteria": "",
        "llm": llm
    }

    return LangChainStringEvaluator(
        "criteria", config=config
    )


def get_context_recall_evaluator(llm: OllamaLLM,
                                 prepare_data: Optional[Callable[[Run, Optional[Example]],
                                 SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    config = {
        "criteria": "",
        "llm": llm
    }

    return LangChainStringEvaluator(
        "criteria",
        config=config,
        prepare_data=prepare_data
    )


def get_context_precision_evaluator(llm: OllamaLLM,
                                    prepare_data: Optional[Callable[[Run, Optional[Example]],
                                    SingleEvaluatorInput]] = None) -> LangChainStringEvaluator:
    config = {
        "criteria": "",
        "llm": llm
    }

    return LangChainStringEvaluator(
        "criteria",
        config=config,
        prepare_data=prepare_data
    )
