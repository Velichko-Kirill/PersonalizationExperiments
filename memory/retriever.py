import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama
from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langsmith import traceable

from init_module.prompts import chain_of_facts_prompt
from memory.consts import ELASTIC_URL

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


#######################################################################################################################
################################## FAISS way to create simulacra's memory #############################################
#######################################################################################################################


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def setup_faiss_retriever(
        memories: List[str],
        embeddings: OllamaEmbeddings,
        num_relevant_docs: int = 4,
        search_type: str = "similarity"
) -> VectorStoreRetriever:
    """Retrieval chain builder"""

    docs = []
    for memory in memories:
        if hasattr(memory, "metadata"):
            metadata = memory.metadata.to_dict()

        else:
            metadata = {}

        docs.append(Document(
            page_content=memory,
            metadata=metadata,
        ))

    store = FAISS.from_documents(docs, embedding=embeddings)
    retriever = store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": num_relevant_docs,
        }
    )
    return retriever


def preprocess_memory(memory: str) -> str:
    mem = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                 '', memory)
    mem = re.sub("<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>", '', mem)
    mem = re.sub("\\n", "", mem)
    mem = re.sub(r"^\d+\.\s+", "", mem)

    return mem


def generate_memories(
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

    return [preprocess_memory(mem)
            for mem in memories if preprocess_memory(mem)]


#######################################################################################################################
################################## Elastic way to create simulacra's memory ###########################################
#######################################################################################################################

class SimulacraMemoryStorage(object):
    def __init__(self, docs: List[Union[str, Document]],
                 index_name: Optional[str] = "simulacra_memory") -> None:
        logger.info("Initialize memory...")
        load_dotenv()
        self.nodes = [
            Document(page_content=doc,
                     metadata={"index": index_name})
            for doc in docs
        ]

        self.embedding_model = OllamaEmbeddings(model="llama3.1")

        self.vector_store = ElasticsearchStore(
            es_url=ELASTIC_URL,
            index_name="simulacra_memory",
            embedding=self.embedding_model,
            distance_strategy="COSINE"
        )

        logger.info("Fill retriever's storage")
        self.retriever = self._get_langchain_retriever()

    def _get_langchain_retriever(self) -> TimeWeightedVectorStoreRetriever:
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store,
            k=3,
            search_kwargs={"num_candidates": 10}
        )

        retriever.add_documents(self.nodes)
        logger.info(f"Memory initialized: {len(retriever.memory_stream)}")

        return retriever

    async def search(self, query: str) -> List[Document]:
        return await self.retriever.ainvoke(query)
