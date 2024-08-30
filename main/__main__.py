import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from pprint import pformat, pprint

import asyncclick as click
from aioconsole import AsynchronousCli
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from init_module.person import Simulacra

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


async def interact_simulacra(simulacra: Simulacra):
    while True:
        inputs = input("observation:")
        reaction = await simulacra.react(inputs)
        print(f"reaction: {reaction}")

        invoked_memories = await simulacra.get_relevant_memories(query=inputs)
        print(f"\nWhat I remember: {pformat(invoked_memories)}")


@click.command()
@click.option("--personal-path", "-p", default="init_module/persons/profiles_MBTI_approach/person_1",
              help="Path to directory containing profile.json")
@click.option("--observations-path", "-o", default="main/observations.txt",
              help="Path to file containing current observations, received from environment or other agents.")
@click.option("--news-path", "-n", default="docs/news/news_moscow.json",
              help="Path to json file containing actual local news")
@click.option("--core-model", "-llm", default="llama3.1",
              help="Core LLM name (id)")
@click.option("--interactive", "-i", default=False,
              help="interactive mode after simulacra's initialization")
async def main(personal_path: Path,
               observations_path,
               news_path: Path,
               core_model: ChatOllama,
               interactive: bool):
    load_dotenv()
    if core_model in ["llama3", "llama3.1"]:

        llm = ChatOllama(model=core_model)

    else:
        raise NotImplementedError(f"Core LLM '{core_model}' is not supported. Use one of Ollama's supported models:\n"
                                  f"https://ollama.com/library")

    # personal_path = Path(os.getcwd(), "init_module", "profiles_MBTI_approach", "person_0")
    simulacra = Simulacra(personal_path, core_llm=llm)

    logger.info(f"Simulacra '{simulacra.name}' initialized")
    logger.info(f"Getting observations from {observations_path}...")

    with open(observations_path, "r") as f:
        observations = f.read().split("###")

    test_data, test_news_processing = [], []

    with open(news_path, "r") as fp:
        news = json.loads(fp.read())

    new_memories = []

    outs_path = Path(personal_path, "outputs")
    os.makedirs(outs_path, exist_ok=True)

    for event_frame in news:
        event = event_frame["mappings"]["summary_detail"]
        assessment = await simulacra.emo_agent_assessment(event=event)
        personalized_event = await simulacra.personalize_observation(event)
        personalized_summary = await simulacra.summarize_obs(personalized_event)

        data = {
            "tag": "[NEWS]",
            "date": event_frame["mappings"]["published"],
            "event": event,
            "assessment": assessment,
            "personalized_event": personalized_event,
            "summary": personalized_summary
        }

        test_news_processing.append(data)
        logger.info(pformat(data))

        new_memories.append(personalized_event)

    with open(Path(outs_path, "news_processing.json"), "w+") as outfile:
        json.dump(test_news_processing, outfile)

    for observation in observations:
        reaction = await simulacra.react(observation)
        personalized_observation = await simulacra.personalize_observation(
            observation=observation
        )
        short_personalized_observation = await simulacra.summarize_obs(personalized_observation)
        new_memories.append(personalized_observation)

        data = {
            "tag": "[LIFE]",
            "observation": observation,
            "reaction": reaction,
            "personalized_observation": personalized_observation,
            "summary": short_personalized_observation
        }

        with open(Path(outs_path, "reactions.json"), "w+") as outfile:
            json.dump(test_data, outfile)

        logger.info(pformat(data))
        test_data.append(data)

    # query_memories = {}
    # for quest in _QUESTS:
    #     memories = await simulacra.get_relevant_memories(query=quest)
    #     query_memories.update({quest: memories})

    # logger.info(pformat(query_memories))
    # await eval_rag(simulacra=simulacra)

    logger.info("Adding new memories...")

    for memory in new_memories:
        await simulacra.add_memory(memory)

    if interactive is False:
        logger.info("Done")
        exit(0)

    else:
        cli = AsynchronousCli()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cli.interact())


if __name__ == '__main__':
    asyncio.run(main())
