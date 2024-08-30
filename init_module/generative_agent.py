import json
import logging
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, Any, Optional, Iterable, Literal, List, Union, TypeVar, Generic

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, RequestError
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_elasticsearch import ElasticsearchStore, ApproxRetrievalStrategy
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_community.vectorstores import FAISS

from init_module.prompts import personality_prompt, chain_of_facts_prompt
from memory import retriever
from memory.prompts import retrieval_prompt
from memory.consts import ELASTIC_URL, VECTORSTORE_MAPPING
from memory.retriever import setup_faiss_retriever, format_docs

"""
    Another way to initialize Simulacra using Elastic vector store.
    Could be built seems like https://github.com/joonspk-research/generative_agents    
"""

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

# EMBEDDING_MODEL_ID: str = "llama3.1"  # dims = 4096
EMBEDDING_MODEL_ID: str = "nomic-embed-text"  # dims = 768


# check memory.requests_n_urls.VECTORSTORE_MAPPING

def es_create_index_if_not_exists(es, index):
    """Create the given ElasticSearch index and ignore error if it already exists"""

    try:
        es.indices.create(index=index, mappings=VECTORSTORE_MAPPING)
        # for debug purposes:
        # mapping = es.indices.get_mapping
        # logger.debug(pformat(mapping))

    except RequestError as ex:
        if ex.error == 'resource_already_exists_exception':
            logger.warning(f"Trying to create existing elasticsearch index '{index}'")
        else:
            raise ex


class Simulacra(GenerativeAgent):
    def __init__(
            self,
            llm: Union[OllamaLLM, ChatOllama, BaseLanguageModel],
            personal_path: Path,
            use_internal_memory: bool = True,
            num_relevant_docs: int = 4,
            verbose: Optional[bool] = False,
            **kwargs
    ) -> None:

        with open(Path(personal_path, "profile.json"), "rb") as f:
            profile: Dict[str, Any] = json.load(f)

        status: str = profile["unique_quality"]
        with open(Path(personal_path, "brief_biography.txt"), "r") as f:
            brief_biography = f.read()

        personality_chain = personality_prompt | llm

        traits = personality_chain.invoke({
            "biography": brief_biography,
            "profile": profile
        })

        if use_internal_memory is False:
            # type(trains) str --> AIMessage()
            traits = traits.content

        memories = self.generate_memories(
            personal_path_=personal_path,
            llm=llm,
        )

        logger.info(f"\n\nBrief description of the simulacra initializing:\n{traits}\n\n"
                    "***************************************************************\n\n")

        memory = GenerativeAgentMemory(
            llm=llm,
            memory_retriever=self._get_retriever(
                memories=memories,
                use_internal_memory=use_internal_memory,
                num_relevant_docs=num_relevant_docs,
            ),
            verbose=True
        )

        super().__init__(
            name=profile["name"],
            age=profile["age"],
            llm=llm,
            status=status,
            memory=memory,
            traits=traits,
            verbose=verbose,
            **kwargs
        )

        self._initialize_memory(
            memories=memories
        )

    # @property
    # def faiss_rag_chain(self) -> RunnableSerializable:
    #     return setup_rag_chain()

    @classmethod
    def preprocess_memory(cls, memory: str) -> str:
        mem = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                     '', memory)
        mem = re.sub("<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", '', mem)
        mem = re.sub("\\n", "", mem)
        mem = re.sub(r"^\d+\.\s+", "", mem)

        return mem

    @classmethod
    def generate_memories(
            cls,
            personal_path_: Path,
            llm: Union[OllamaLLM, ChatOllama, BaseLanguageModel],
    ) -> List[str]:
        chunks = pd.read_csv(Path(personal_path_, "chunks.csv"))["chunk"]
        memories = []
        memory_details_chain = chain_of_facts_prompt | llm

        for chunk in chunks:
            memories_part_str = memory_details_chain.invoke({
                "biography_part": chunk
            }).content

            memoires_part = memories_part_str.split("\n")
            memories.extend(memoires_part)

        return [cls.preprocess_memory(mem)
                for mem in memories]

    def _initialize_memory(self, memories: List[str]) -> None:

        # logger.debug(pformat(memories))
        # docs = []
        # For external facts in memory remove all links and html tags

        for i, mem in enumerate(memories[:20]):
            if mem:
                mem = self.preprocess_memory(mem)
                self.memory.add_memory(mem)

            else:
                logger.warning("Trying to add an empty memory")

        # logger.info(f"Added {len(docs)} memories")

        all_mems_path = Path("results", "all_memories.json")
        with open(all_mems_path, "w") as f:
            json.dump(memories, f)

        logger.info(f"Primary memories added to {self.name}. "
                    f"Its can be found at: {str(all_mems_path)}\n")

    def _get_retriever(self, use_internal_memory: bool,
                       memories: List[str],
                       num_relevant_docs: int = 4):
        if use_internal_memory is True:
            memory_retriever: TimeWeightedVectorStoreRetriever = TimeWeightedVectorStoreRetriever(
                vectorstore=self.vector_store,
                other_score_keys=["importance", "relevancy"],
                k=num_relevant_docs,
                search_kwargs={"num_candidates": 10}
            )

        elif use_internal_memory is False:
            memory_retriever: VectorStoreRetriever = setup_faiss_retriever(
                memories=memories,
                num_relevant_docs=num_relevant_docs,
                embeddings=self.embedding_model,
            )

        else:
            raise TypeError("Impossible to define memory type, set `use_internal_memory`(bool) var")

        return memory_retriever

    @property
    def embedding_model(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(model=EMBEDDING_MODEL_ID)

    @property
    def vector_store(self) -> ElasticsearchStore:
        es_conn = Elasticsearch(ELASTIC_URL)

        time = datetime.today().strftime("%Y%m%d%H%M%S")
        idx = f"simulacra_memory_{time}"
        es_create_index_if_not_exists(es_conn, idx)

        return ElasticsearchStore(
            es_connection=es_conn,
            index_name=idx,
            embedding=self.embedding_model,
            strategy=ApproxRetrievalStrategy(),
            distance_strategy="COSINE"
        )

    @property
    def retriever(self) -> Union[TimeWeightedVectorStoreRetriever, FAISS]:
        return self.memory.memory_retriever

    async def answer(self, query: str,
                     num_relevant_documents: int = 4) -> List[str]:

        docs = self.memory.fetch_memories(observation=query,
                                          now=datetime.now())
        return [d.page_content for d in docs]

        # # question_emb = await self.embedding_model.aembed_query(question)
        # # vec = np.array(question_emb)
        # # logger.info(vec.shape)
        # # relevant_docs = self.retriever.get_relevant_documents(question)
        #
        # relevant_docs = await self.vector_store.asimilarity_search(query=query)
        # logger.info(f"Relevant docs: {pformat(
        #     [doc.page_content for doc in relevant_docs]
        # )}")
        #
        # # answers = await self.retriever.ainvoke(query)
        # answers = await self.retriever.ainvoke(query)

        #
        # return [doc.page_content for doc in answers]
