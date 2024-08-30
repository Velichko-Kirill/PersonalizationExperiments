import asyncio
import contextlib
import json
import logging
import os
import sys
from pathlib import Path
from pprint import pformat, pprint

import asyncclick as click
from aioconsole import ainput, AsynchronousCli
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from init_module.person import Simulacra

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


async def interact_simulacra(simulacra: Simulacra):
    print("Interactive mode: Type 'exit' to quit.")
    print("You can enter 'observe <your_observation>' to process a new observation.")
    print("You can enter 'news <your_news_event>' to process a new news event.")

    while True:
        user_input = await ainput(">>> ")

        if user_input.lower() == "exit":
            print("Exiting interactive mode.")
            break

        elif user_input.startswith("observe "):
            observation = user_input[len("observe "):]
            reaction = await simulacra.react(observation)
            personalized_observation = await simulacra.personalize_observation(observation)
            short_personalized_observation = await simulacra.summarize_obs(personalized_observation)

            print(f"Reaction: {reaction}")
            print(f"Personalized Observation: {personalized_observation}")
            print(f"Summary: {short_personalized_observation}")

            # Optionally, you could save this to the memory
            await simulacra.add_memory(personalized_observation)

        elif user_input.startswith("news "):
            news_event = user_input[len("news "):]
            assessment = await simulacra.emo_agent_assessment(news_event)
            personalized_event = await simulacra.personalize_observation(news_event)
            personalized_summary = await simulacra.summarize_obs(personalized_event)

            print(f"Assessment: {assessment}")
            print(f"Personalized Event: {personalized_event}")
            print(f"Summary: {personalized_summary}")

            # Optionally, you could save this to the memory
            await simulacra.add_memory(personalized_event)

        else:
            reaction = await simulacra.react(user_input)
            print(reaction)

            personalized_event = await simulacra.personalize_observation(user_input)
            await simulacra.add_memory(personalized_event)
            print(f"Added a memory: \n{personalized_event}")


@click.command()
@click.option("--person-id", "-id", default="0",
              help="Simulacra's id, used for getting path to directory containing profile.json")
@click.option("--observations-path", "-o", default="main/observations.txt",
              help="Path to file containing current observations, received from environment or other agents.")
@click.option("--news-path", "-n", default="docs/news/news_moscow.json",
              help="Path to json file containing actual local news")
@click.option("--core-model", "-llm", default="llama3.1",
              help="Core LLM name (id)")
@click.option("--interactive", "-i", is_flag=True, default=False,
              help="interactive mode after simulacra's initialization")
async def main(person_id: Path,
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
    personal_path = Path(os.getcwd(), "init_module", "persons",
                         "profiles_MBTI_approach", f"person_{person_id}")

    simulacra = Simulacra(personal_path, core_llm=llm)

    logger.info(f"Simulacra '{simulacra.name}' initialized")

    if interactive is True:
        # with contextlib.suppress(LangSmithRateLimitError):
        await interact_simulacra(simulacra=simulacra)

        exit(0)

    logger.info(f"Getting observations from {observations_path}...")

    with open(observations_path, "r") as f:
        observations = f.read().split("###")

    observations_reactions, test_news_processing = [], []

    with open(news_path, "r") as fp:
        news = json.loads(fp.read())

    new_memories = []

    outs_path = Path(personal_path, "outputs")
    os.makedirs(outs_path, exist_ok=True)

    for event_frame in news[:5]:
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

    with open(Path(outs_path, "news_processing.json"), "w") as outfile:
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

        logger.info(pformat(data))
        observations_reactions.append(data)

    with open(Path(outs_path, "reactions.json"), "w") as outfile:
        json.dump(observations_reactions, outfile)

    logger.info("Adding new memories...")

    for memory in new_memories:
        await simulacra.add_memory(memory)

    # with open(Path(personal_path, "memories.json"), "w") as outfile:
    #     json.dump(new_memories, outfile)

    if interactive is False:
        logger.info("Done")
        exit(0)

    else:
        await interact_simulacra(simulacra)
        return


if __name__ == '__main__':
    asyncio.run(main())
