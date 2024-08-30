import itertools
import json
import logging
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from pprint import pformat, pprint
from typing import Optional, Dict, Union

import asyncclick as click

# Currently, cannot be imported with python 3.12
# from datasets import load_dataset, concatenate_datasets

from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.chains.base import Chain
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_experimental.llms import ChatLlamaAPI
from langchain_ollama import ChatOllama

import nltk
import pandas as pd
from transformers import MBartTokenizer, MBartForConditionalGeneration, GenerationConfig

from .estimators import (ImportanceEstimator,
                         RedundancyEstimator)
from .chains import (
    build_primary_chain,
    build_psycho_desc_chain,
    build_enrichment_chain,
)

from init_module.person import Profile
from .prompts import (high_level_bio_prompt,
                      expand_chunk_prompt, psycho_desc_prompt, )

from .consts import (SEP_TOKEN,
                     NUM_SAMPLES,
                     SUMMARIZER_ID, NUM_ATTEMPTS)
from worldview.consts import PERSONAL_TRAITS, MBTI_LABELS

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


# TODO decorator for attempts

def generate_primary_bio(model: [ChatLlamaAPI],
                         person_: Profile,
                         attempts_: Optional[int] = 1) -> str:
    """Generating the high-level biography of the Person specified"""

    primary_bio_chain = build_primary_chain(template=high_level_bio_prompt,
                                            model=model)
    primary_bio_ = ""

    for attempt in range(attempts_):

        try:
            primary_bio_ = primary_bio_chain.invoke({"traits": person_}).content

        except json.decoder.JSONDecodeError as jsonDecEx:
            logger.warning(f"Can't decode {type(person_).__name__}!\n{pformat(jsonDecEx)}\n"
                           f"(attempt {attempt + 1}/{attempts_}):\n")

            if attempt == attempts_ - 1:
                raise TimeoutError(f"Maximum number of attempts ({attempts_}) reached!")

        except Exception as ex:
            logger.warning(f"Unexpected exception while generating the primary bio:\n\n{pformat(ex)} Skipping loop...")
            traceback.print_exc()
            continue

    return primary_bio_


############################### By the simple prompt approach #####################################
# def summarize(chain: Chain, text: str, num_words: int,
#               attempts_: Optional[int] = 5) -> str:
#     summarized_text = ""
#     chain = chain.with_types(input_type=SummaryRequest)
#     # summary_request = SummaryRequest(text=text,
#     #                                  num_words=num_words)
#
#     for attempt in range(attempts_):
#         try:
#             text = re.sub("\"", "", text)
#             summarized_text = chain.invoke({"text": str(text),
#                                             "num_words": int(num_words)}).content
#
#         except json.decoder.JSONDecodeError as jsonDecEx:
#             logger.warning(f"Can't decode text: {text}!"
#                            f"(attempt {attempt + 1}/{attempts_}):\n")
#
#             if attempt == attempts_ - 1:
#                 raise TimeoutError(f"Maximum number of attempts ({attempts_}) reached!")
#
#         else:
#             break
#
#     return summarized_text


def summarize(text: str, num_words: Optional[int] = None) -> str:
    """All the summarization settings should be given right here in `generation_config`"""

    tokenizer = MBartTokenizer.from_pretrained(SUMMARIZER_ID)
    model = MBartForConditionalGeneration.from_pretrained(SUMMARIZER_ID)
    if num_words is None:
        num_words = len(text.split())

    generation_config = GenerationConfig(
        max_new_tokens=num_words
    )

    input_ids = tokenizer(
        [text],
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4,
        generation_config=generation_config
    )[0]

    summary = tokenizer.decode(output_ids, skip_special_tokens=True)

    return summary


async def enrich(chain: Chain,
                 chunk_: str,
                 importance_: float,
                 redundancy_: float,
                 psychological: str,
                 attempts_: Optional[int] = 5) -> str:
    # summary_request = SummaryRequest(text=text,
    #                                  num_words=num_words)
    enriched_chunk_ = chunk_

    for attempt in range(attempts_):
        try:
            enriched_chunk_ = chain.invoke({
                "chunk": str(chunk_),
                "importance": importance_,
                "redundancy": redundancy_,
                "psychological": psychological,
            }).content

        except json.decoder.JSONDecodeError as jsonDecEx:
            logger.warning(f"Can't decode text: {chunk_}!"
                           f"(attempt {attempt + 1}/{attempts_}):\n")
            logger.error(pformat(jsonDecEx))

            if attempt == attempts_ - 1:
                raise TimeoutError(f"Maximum number of attempts ({attempts_}) reached!")

        else:
            break

    return enriched_chunk_


def get_birthday(age: int) -> str:
    current_datetime = datetime.now()
    birth_year = current_datetime.year - age
    birth_month = random.randint(1, current_datetime.month)
    birth_day = random.randint(1, current_datetime.day)

    return f"{birth_year}.{birth_month}.{birth_day}"


def get_age_by_birthday(birthday: str) -> int:
    current_datetime = datetime.now()
    logger.debug(f"birthday is {birthday}")
    logger.debug(f"current_datetime: {current_datetime}")


def set_personal_traits(row: Union[Dict[str, float], pd.Series]):
    traits = {
        trait: row[trait]
        for trait in PERSONAL_TRAITS
    }

    logger.info(traits)

    return traits


async def get_summary_dict(profile) -> Dict[str, str]:
    try:
        mbti_type = profile.personal_traits["MBTI"]

    except KeyError:
        logger.error("MBTI profiling don't using, choose another way to get agent's summary")
        return {"Unknown": "Unknown"}

    path_to_mbti_description: Path = Path().resolve() / "docs" / "personality_tests" / "mbti_characteristics_en.xlsx"
    df = pd.read_excel(path_to_mbti_description, index_col=0)
    row_data = df.loc[mbti_type]
    summary_dict = row_data.to_dict()

    return summary_dict


# TODO use a wrapper instead, like in questionnaire.py
async def generate_advanced_bio(
        personal_path: Path,
        model: ChatOllama,
        attempts: Optional[int] = 5,
        sep: Optional[str] = SEP_TOKEN,
) -> str:
    with open(Path(personal_path) / "profile.json", "r") as config_file:
        profile = Profile(**json.load(config_file))

    # simulacra = Simulacra(
    #     personal_path,
    #     core_llm=model,
    #     initialize=True,
    # )

    if "MBTI" in profile.personal_traits.keys():
        mbti_label = profile.personal_traits["MBTI"]

        assert mbti_label in MBTI_LABELS, (f"MBTI mode is on, but label must be one of: {pformat(MBTI_LABELS)}\n"
                                           f"'{mbti_label}' instead")

        # reflection_agent = ReflectionAgent()
        psycho_chain = build_psycho_desc_chain(template=psycho_desc_prompt,
                                               model=model)
        psychological_description = psycho_chain.invoke({
            "psycho_profile": await get_summary_dict(profile=profile)
        }).content

        profile.personal_traits["description"] = psychological_description

    else:
        logger.warning(f"Personality undefined!\n{pformat(profile.personal_traits)}")
        psychological_description = ""

    primary_bio = generate_primary_bio(model=model,
                                       person_=profile,
                                       attempts_=attempts)

    # TODO use langchain.text_splitters ?

    chunks = primary_bio.split(sep)
    logger.debug(f"{len(chunks)} number of chunks")

    words = list(itertools.chain.from_iterable([
        chunk.split() for chunk in chunks if chunk
    ]))

    mean_n_words = len(words) // len(chunks)
    logger.info(f"we got {mean_n_words} mean number of words in chunks provided")
    # summarize_chain = build_summarize_chain(template=summarize_prompt,
    #                                         model=model)
    #
    # essential = summarize(chain=summarize_chain,
    #                       text=primary_bio,
    #                       num_words=mean_n_words)

    essential = summarize(text=primary_bio,
                          num_words=mean_n_words)
    backbone = SEP_TOKEN.join(chunks)
    logger.info(pformat(backbone))

    importance_estimator = ImportanceEstimator()
    redundancy_estimator = RedundancyEstimator(
        summarize_func=summarize,
        # if use a prompting way 4 summarization
        # summarize_func=partial(
        # summarize, summarize_chain)
        # **{"device": torch.device("cpu")}
    )

    enrichment_chain = build_enrichment_chain(template=expand_chunk_prompt,
                                              model=model)

    sents = nltk.sent_tokenize(primary_bio)
    mean_sent_words = len(words) // len(sents)
    sims, reds = [], []

    enriched_bio = ""
    scored_chunks = pd.DataFrame(columns=["chunk", "importance", "redundancy"])

    for i, chunk in enumerate(chunks):
        if chunk:
            importance = importance_estimator(essential, chunk)
            redundancy = redundancy_estimator(chunk,
                                              num_words=mean_sent_words)

            logger.debug(f"chunk # {i + 1}: {pformat(chunk)}\n"
                         f"importance: {str(importance)}\n"
                         f"redundancy: {str(redundancy)}\n"
                         f"psychological: {psychological_description}"
                         )

            sims.append(importance)
            reds.append(redundancy)

            expanded_chunk = await enrich(chain=enrichment_chain,
                                          chunk_=chunk,
                                          importance_=importance,
                                          redundancy_=redundancy,
                                          psychological=psychological_description,
                                          attempts_=attempts)

            enriched_bio = SEP_TOKEN.join([enriched_bio, expanded_chunk])

    os.makedirs(personal_path, exist_ok=True)

    scored_chunks["chunk"] = chunks
    scored_chunks["importance"] = sims
    scored_chunks["redundancy"] = reds

    with open(os.path.join(personal_path, "enriched_biography.txt"), "w") as fp:
        fp.write(enriched_bio)

    logger.info(f"Saved in {personal_path}")

    with open(os.path.join(personal_path, "brief_biography.txt"), "w") as fp:
        fp.write(essential)

    with open(os.path.join(personal_path, "profile.json"), "w") as fp:
        json.dump(profile.__dict__, fp)

    scored_chunks.to_csv(os.path.join(personal_path, "chunks.csv"))

    return enriched_bio


def generate_samples(profiles_dir, model):
    profiles_path = os.path.join(profiles_dir,
                                 "syntetic_profiles.json")
    data = []

    with open(profiles_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    with open(os.path.join(profiles_dir, "mapper.json"), "r") as fp:
        mapper = json.load(fp)

    df = df.rename(columns=mapper)
    personality_ds = load_dataset("Fatima0923/Automated-Personality-Prediction")
    # print(type(personality_ds))
    # print(personality_ds.keys())
    ds = concatenate_datasets([personality_ds["train"],
                               personality_ds["validation"],
                               personality_ds["test"], ])
    ds_part = list(df.iterrows())[27:]

    for id_, profile_series in ds_part:
        profile = profile_series.to_dict()
        profile["id"] = id_
        profile["birthday"] = get_birthday(age=profile["age"])
        print(profile["birthday"])
        try:
            ds_row = ds[id_]

        except IndexError:
            logger.error("Dataset overflow. Skipping loop...")
            continue

        profile["personal_traits"] = set_personal_traits(row=ds_row)
        _ = profile.pop("Story")

        try:
            generate_advanced_bio(profile=profile, model=model,
                                  attempts=10, sep=SEP_TOKEN)

        except TimeoutError as err:
            logger.error(err)
            continue

        if id_ >= NUM_SAMPLES:
            logger.info(f"{NUM_SAMPLES} samples successfully reached. Finish generation.")


@click.command()
@click.option("--personal_path", "-p", default="init_module/persons/profiles_MBTI_approach/person_0")
async def main(personal_path):
    load_dotenv()
    nltk.download('punkt')
    llama: ChatOllama = ChatOllama(model="llama3.1")

    # for person_id in range(16, 32):
    #     personal_path: Path = Path(
    #         __file__).parent.parent / "init_module" / "profiles_MBTI_approach" / f"person_{person_id}"
    biography = await generate_advanced_bio(personal_path,
                                            model=llama,
                                            attempts=NUM_ATTEMPTS)

    pprint(biography)
    print("\n\n", "*" * 40, "\n\n")


if __name__ == '__main__':
    main(_anyio_backend="asyncio")
