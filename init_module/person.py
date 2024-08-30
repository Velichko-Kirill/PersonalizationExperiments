import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint, pformat
from typing import Any, List, Dict, Union

from langchain_core.documents import Document
from pydantic import BaseModel, Field
import pandas as pd

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig, RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaLLM, OllamaEmbeddings
from transformers import MBartForConditionalGeneration, MBartTokenizer

from init_module.prompts import personality_prompt, role_system_prompt, emo_agent_prompt, personal_perception_prompt, \
    summarization_prompt
from memory.consts import SUMMARIZER_ID
from memory.prompts import retrieval_prompt, qa_prompt
from memory.retriever import format_docs, generate_memories, setup_faiss_retriever
from scheduler.scheduler import Scheduler

# EMBEDDING_MODEL_ID: str = "llama3.1"  # dims = 4096
EMBEDDING_MODEL_ID: str = "nomic-embed-text"  # dims = 768

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


class Profile(BaseModel):
    id: int = Field(description="Unique identifier of simulacra")
    name: str = Field(description="Full name of the person")
    birthday: str = Field(description="Birthday")
    hobbies: List[str] = Field(description="Hobbies", default=["Unknown"])
    personal_traits: Dict[str, Union[str, float]] = Field(description="psychological personality traits",
                                                          default={"MBTI": "Unknown"})
    family: Dict[str, Any] = Field(description="Family of the person", default={"family": "single"})

    unique_quality: str = Field(description="Unique quality", default="Unknown")

    education: Dict[str, str] = Field(description="all schools and educational institutions",
                                      default={
                                          "School": "Unknown"
                                      })

    career: Dict[str, str] = Field(description="Current job",
                                   default={
                                       "Company": "Unknown",
                                       "position": "Unknown"
                                   })

    residence: str = Field(description="Current residence. May differ from homeland. "
                                       "Should be realized as an instance of `Country` class",
                           default="Unknown")


class Simulacra:

    def __init__(self, personal_path, /,
                 core_llm: ChatOllama,
                 personal_type: str = "balanced",
                 init_memory: bool = True,
                 init_scheduler: bool = True,
                 **kwargs: Any):

        # assert personal_path.exists(), f"Wrong personal path: {personal_path}"

        with open(Path(personal_path, "profile.json"), "rb") as f:
            # for validation purposes
            profile_json = json.load(f)

        print(F"personal_path: {personal_path}")
        pprint(profile_json)
        self.profile: Profile = Profile(**profile_json)

        self.personal_path = personal_path
        logger.info(f"Simulacra '{self.profile.name}' is initializing from directory:\n{personal_path}\n")
        self.mem_path = Path(personal_path, "memories.json")

        self.llm = core_llm
        logger.info(f"Core LLM: {self.llm.__class__.__name__}")

        for key, value in self.profile.model_dump().items():
            setattr(self, key, value)

        with open(Path(personal_path, "brief_biography.txt"), "r") as f:
            self.brief_biography = f.read()

        if init_memory is True:
            # memory setup
            num_memories = kwargs.get("num_memories")
            if num_memories is None:
                num_memories = 4  # default k value in GenerativeAgents

            self.num_memories = num_memories

            memories = generate_memories(personal_path, llm=core_llm)
            logger.debug(pformat(memories))

            with open(self.mem_path, "w") as fp:
                json.dump(memories, fp=fp, indent=4)

            self.retriever: VectorStoreRetriever = setup_faiss_retriever(
                memories=memories,
                num_relevant_docs=self.num_memories,
                embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL_ID),
            )
            self.rag_chain = (
                    {
                        "context": self.retriever | format_docs,
                        "input": RunnablePassthrough(),
                    }
                    | retrieval_prompt
                    | core_llm
                    | StrOutputParser()
            )

        if init_scheduler is True:
            input_file = Path().resolve().parent / "scheduler" / f"{personal_type}.txt"
            logger.debug(f"Scheduler template file: {input_file}")

            self.scheduler = Scheduler(personal_path=personal_path,
                                       rag_chain=self.rag_chain,
                                       template_file=input_file,
                                       llm=self.llm)

        # setup all basic chains
        self.qa_chain = qa_prompt | self.llm
        self.reaction_chain = role_system_prompt | self.llm
        self.summ_chain = summarization_prompt | self.llm
        self.personality_chain = personality_prompt | self.llm
        self.emo_chain = emo_agent_prompt | self.llm
        self.personal_perception_chain = personal_perception_prompt | self.llm

    async def summarize_obs(self, observation: str) -> str:
        psy_summary = await self.get_summary()
        summary = await self.summ_chain.ainvoke({
            "observation": observation,
            "summary": psy_summary,
        })

        return summary.content

    async def fill_schedule(self, start_date: str):
        self.scheduler.init_schedule(start_date)
        await self.scheduler.fill_schedule()

        path_to_schedule = self.personal_path / f"{self.name}_schedule.csv"
        self.scheduler.schedule_to_csv(path_to_schedule)

        logger.info(f"Scheduler for person '{self.name}' successfully created and available at:\n{path_to_schedule}\n")

    async def get_relevant_memories(self, query: str, **kwargs: Any) -> List[str]:
        return await self.retriever.aget_relevant_documents(query, **kwargs)

    async def answer(self, question: str) -> str:
        """Using QA prompt for testing purposes"""
        contexts = await self.get_relevant_memories(query=question)

        return await self.qa_chain.ainvoke({
            "question": question,
            "context": contexts
        }).content

    async def get_summary_dict(self) -> Dict[str, str]:

        try:
            mbti_type = self.profile.personal_traits["MBTI"]

        except KeyError:
            logger.error("MBTI profiling don't using, choose another way to get agent's summary")
            return {"Unknown": "Unknown"}

        path_to_mbti_description: Path = Path().resolve() / "docs" / "personality_tests" / "mbti_characteristics_en.xlsx"
        df = pd.read_excel(path_to_mbti_description, index_col=0)
        data = df.loc[mbti_type]
        summary_dict = data.to_dict()

        return summary_dict

    async def get_summary(self) -> str:
        psycho_profile = await self.get_summary_dict()

        summary_msg = await self.personality_chain.ainvoke({
            "biography": self.brief_biography,
            "profile": psycho_profile
        })

        return summary_msg.content

    async def react(self, observation: str) -> str:
        summary = await self.get_summary_dict()

        reaction = await self.reaction_chain.ainvoke({
            "summary": summary,
            "observation": observation
        })

        logger.info(f"Reaction: {reaction}")

        return reaction.content

    async def emo_agent_assessment(self, event: str) -> str:
        """Emotional agent"""

        summary = await self.get_summary_dict()
        emotional_estim_msg = await self.emo_chain.ainvoke({
            "summary": summary,
            "biography": self.brief_biography,
            "event": event
        })

        emotional_estim = emotional_estim_msg.content

        return emotional_estim

    async def personalize_observation(self, observation: str) -> str:
        """Personalization new events for reaction and memorization"""

        emo_assessment = await self.emo_agent_assessment(observation)
        observation_rewrote_msg = await self.personal_perception_chain.ainvoke(
            {"observation": observation,
             "assessment": emo_assessment})

        observation_rewrote = observation_rewrote_msg.content

        return observation_rewrote

    async def add_memory(self, memory: str) -> None:
        doc = Document(page_content=memory,
                       metadata={"datetime": datetime.now()})
        await self.retriever.aadd_documents([doc])

        with open(self.mem_path, "w+") as fp:
            json.dump(memory, fp=fp, indent=4)

        logger.info(f"Memory added: {memory}")
