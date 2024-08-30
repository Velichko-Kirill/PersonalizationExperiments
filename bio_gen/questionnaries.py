from functools import wraps
import json
import logging
import os
import sys
from collections import defaultdict
from pprint import pformat, pprint
from typing import Callable

from docx import Document
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from bio_gen.prompts import answer_prompt
from init_module.prompts import role_system_prompt

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


def get_bio(personal_path):
    with open(os.path.join(personal_path, "brief_biography.txt")) as fp:
        return fp.read()


def several_attempts(func: Callable, num_attempts: int = 3) -> Callable:
    """API requests wrapper for a several parsing attempts"""

    @wraps(func)
    def attempts_wrapper(chain, utterance, description, *args, **kwargs):

        score = 0
        for attempt in range(num_attempts):
            try:
                score = func(chain, utterance, description, *args, **kwargs)

            except TypeError as type_err:
                logger.info(f"attempt # {attempt + 1}")
                logger.info(f"The utterance `{utterance}` wasn.t proceed.\n"
                            f"traceback: {pformat(type_err)}\n")

            else:
                logger.info(f"attempt # {attempt + 1}. Success")
                break

        return score

    return attempts_wrapper


@several_attempts
def proceed_question(chain: LLMChain, description: str, utterance: str) -> int:
    score = chain.invoke({
        "description": description,
        "utterance": utterance
    })
    logger.debug(f"score: {score}")

    try:
        numeric_score = int(score.content)

    except ValueError:
        logger.warning("Wrong string value for a score obtained!"
                       f"\ncontent: {pformat(score.content)}")

        return 0

    return numeric_score


def ask_agent(quest_path, model, personal_path):
    document = Document(quest_path)
    table = document.tables[0]

    cols = [cell.text for cell in table.rows[0].cells]
    logger.info(cols)
    prompt = PromptTemplate(template="\n".join([role_system_prompt, answer_prompt]),
                            input_variables=["utterance", "description"])

    chain = prompt | model
    traits = defaultdict(int)

    description = get_bio(personal_path)

    for row in tqdm(table.rows[1:]):
        vals = list(map(lambda x: x.text, row.cells))
        utterance = vals[1]

        score = proceed_question(chain=chain,
                                 description=description,
                                 utterance=utterance)

        traits[vals[0]] += score
        pprint(traits)
        with open(os.path.join(personal_path, "quest_result.json"), "w") as fp:
            json.dump(traits, fp=fp)

    return traits


