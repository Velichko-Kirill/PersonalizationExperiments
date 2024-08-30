import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
from pprint import pprint
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.llms import ChatLlamaAPI
from langchain_ollama import ChatOllama
from llamaapi import LlamaAPI
from tqdm import tqdm

from init_module.person import Simulacra
from worldview.prompts import quest_1_prompt, quest_2_prompt


@dataclass
class RegexIn:
    string: str

    def __eq__(self, other: str | re.Pattern):
        if isinstance(other, str):
            other = re.compile(other)

        assert isinstance(other, re.Pattern)

        return other.match(self.string) is not None


async def compute_final_type(scores: Dict[str, int], source_type: str) -> str:
    if scores["E"] > scores["I"]:
        l_1 = "E"

    elif scores["I"] > scores["E"]:
        l_1 = "I"

    else:
        l_1 = source_type[0]

    if scores["N"] > scores["S"]:
        l_2 = "N"

    elif scores["S"] > scores["N"]:
        l_2 = "S"

    else:
        l_2 = source_type[1]

    if scores["P"] > scores["F"]:
        l_3 = "P"

    elif scores["F"] > scores["P"]:
        l_3 = "F"

    else:
        l_3 = source_type[2]

    if scores["T"] > scores["J"]:
        l_4 = "T"

    elif scores["J"] > scores["T"]:
        l_4 = "J"

    else:
        l_4 = source_type[3]

    return "".join((l_1, l_2, l_3, l_4))


async def save_scores(scores: Dict[str, Any],
                      simulacra: Simulacra,
                      question_answer: Dict[str, str],
                      postfix: str) -> None:
    source_type = simulacra.profile.personal_traits["MBTI"]
    personal_path = simulacra.personal_path
    summary = await simulacra.get_summary()

    scores.update({"source_type": source_type})
    final_type = await compute_final_type(scores, source_type)

    scores.update({"final_type": final_type})
    scores.update({"description": summary})

    with open(Path(personal_path, f"answers_{postfix}.json"), "w") as f:
        json.dump(scores, f)

    with open(Path(personal_path, f"question_answer_{postfix}.json"), "w") as f:
        json.dump(question_answer, f)

    with open(Path(personal_path, "psycho_summary.txt"), "w") as f:
        f.write(summary)


async def pass_quest_1(
        model: ChatOllama,
        prefix: str,
        num_persons: int = 32,
) -> None:
    quest_1_path = os.path.join(os.getcwd(), "docs", "personality_tests", "questionnaire_1_en.xlsx")
    quest_1 = pd.read_excel(quest_1_path)

    for j in range(num_persons):

        personal_path = Path(prefix, f"person_{j}")
        scores = {
            "E": 0,
            "I": 0,
            "N": 0,
            "S": 0,
            "T": 0,
            "J": 0,
            "P": 0,
            "F": 0,
            "neutral_answers": 0,
            "unexpected_answers": []
        }

        simulacra = Simulacra(personal_path,
                              init_memory=False,
                              init_scheduler=False,
                              core_llm=model)

        quest_1_chain = quest_1_prompt | simulacra.llm
        summary = await simulacra.get_summary()

        print(f"summary: {summary}")
        question_answer = {}

        for i, row in tqdm(quest_1.iterrows()):

            utterance = row["utterance"]
            answer = quest_1_chain.invoke({
                "utterance": utterance,
                "summary": summary,
            }).content

            question_answer.update({utterance: answer})

            agree_key = row["agree_key"]
            disagree_key = row["disagree_key"]

            match RegexIn(answer):

                case r"I? [Cc]ompletely agree\w*":
                    scores[agree_key] += 3

                case r"I? [Rr]ather agree\w*":
                    scores[agree_key] += 2

                case r"I? [Ss]omewhat agree\w*":
                    scores[agree_key] += 1

                case r"I? [Ss]omewhat disagree\w*":
                    scores[disagree_key] += 1

                case r"I? [Rr]ather disagree\w*":
                    scores[disagree_key] += 2

                case r"I? [Cc]ompletely disagree\w*":
                    scores[disagree_key] += 3

                case r"I? [Nn]either agree,? nor disagree\w*":
                    scores["neutral_answers"] += 1

                case _:
                    print(f"unexpected answer: {answer}")

                    exception = {
                        "utterance": utterance,
                        "agree_key": agree_key,
                        "disagree_key": disagree_key,
                        "unexpected_answer": answer
                    }
                    scores["unexpected_answers"].append(exception)

        pprint(scores)

        await save_scores(scores=scores,
                          simulacra=simulacra,
                          question_answer=question_answer,
                          postfix="quest_1")


async def pass_quest_2(model: ChatOllama,
                       prefix: str,
                       num_persons: int = 32) -> None:
    quest_2_path = os.path.join(os.getcwd(), "docs", "personality_tests", "questionnaire_2_en.xlsx")
    quest_2 = pd.read_excel(quest_2_path)

    for j in range(num_persons):
        personal_path = Path(prefix, f"person_{j}")
        scores = {
            "E": 0,
            "I": 0,
            "N": 0,
            "S": 0,
            "T": 0,
            "J": 0,
            "P": 0,
            "F": 0,
            "neutral_answers": 0,
            "unexpected_answers": []
        }

        simulacra = Simulacra(personal_path,
                              init_memory=False,
                              init_scheduler=False,
                              core_llm=model)

        quest_2_chain = quest_2_prompt | simulacra.llm

        summary = await simulacra.get_summary()

        question_answer = {}

        for i, row in tqdm(quest_2.iterrows()):
            utterance = row["utterance"]
            answer_a = row["answer_a"]
            answer_b = row["answer_b"]
            key_a = row["key_a"]
            key_b = row["key_b"]

            answer = quest_2_chain.invoke({
                "utterance": utterance,
                "summary": summary,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }).content

            match RegexIn(answer):
                case r"[Aa]":
                    scores[key_a] += 1

                case r"[Bb]":
                    scores[key_b] += 1

                case _:
                    exception = {
                        "utterance": utterance,
                        "key_a": key_a,
                        "key_b": key_b,
                        "unexpected_answer": answer
                    }
                    scores["unexpected_answers"].append(exception)

            question_answer.update({utterance: answer})

        pprint(scores)

        await save_scores(scores=scores,
                          simulacra=simulacra,
                          question_answer=question_answer,
                          postfix="quest_2")


async def main():
    load_dotenv()
    llm = ChatOllama(model="llama3.1")
    prefix = os.path.join(os.getcwd(), "init_module", "persons", "profiles_MBTI_approach")

    await pass_quest_1(llm, prefix)
    await pass_quest_2(llm, prefix)

    exit(0)


if __name__ == '__main__':
    asyncio.run(main())
