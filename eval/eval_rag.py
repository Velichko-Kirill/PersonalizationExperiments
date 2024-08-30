import asyncio
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from pprint import pformat, pprint
import warnings
from typing import Any, Dict, Callable, Optional

import pandas as pd

from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from langchain_benchmarks import registry
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.vectorstores import ElasticVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, aevaluate
from langsmith.evaluation._runner import ExperimentResults
from ragas.embeddings import LangchainEmbeddingsWrapper
from langsmith.evaluation import evaluate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from tqdm import tqdm

from init_module.person import Simulacra
from memory.prompts import retrieval_prompt, qa_prompt
from memory.consts import ELASTIC_URL
from eval.evaluators import (
    get_correctness_evaluator,
    get_relevancy_evaluator
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

_EVAL_DATASET_ID: str = "explodinggradients/ragas-wikiqa"
registry.filter(Type="RetrievalTask")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def simple_test(agent):
    # Test the memory
    observations = [
        "British scientists discover a new kind of AI-bots",
        "All we need is a new method to safeguard power systems",
        "Gender discrimination is the most important problem over the world"
    ]
    result = []

    for observation in observations:
        docs_found = agent.memory.fetch_memories(observation=observation)
        related_memoires = [doc.page_content for doc in docs_found]
        result.append({
            "observation": observation,
            "related_memoires": related_memoires
        })

        logger.info(f"Related memoires to observation\n'{observation}':\n"
                    f"'{pformat(related_memoires)}'")

    results_path = (Path(__file__).parent / "results" / "results.json")

    with open(results_path, "w") as fp:
        json.dump(result, fp, indent=4)


async def eval_rag_(eval_llm,
                    embeddings,
                    retriever,
                    raw_data: bool = True,
                    df=None):
    if df is None:
        df = load_dataset("explodinggradients/fiqa", "ragas_eval")[
            "baseline"].select(range(3))
        # Small size dataset

    metrics = [
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]
    langchain_llm = LangchainLLMWrapper(eval_llm)
    langchain_embeddings = LangchainEmbeddingsWrapper(embeddings)

    for m in metrics:
        m.__setattr__("llm", langchain_llm)

    retrieval_chain = create_stuff_documents_chain(
        llm=eval_llm,
        prompt=retrieval_prompt
    )

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=retrieval_chain
    )

    outputs = []

    if raw_data is True:
        for _, row in tqdm(df.iterrows(),
                           total=df.shape[0], desc="Processing rows"):
            answer = await qa_chain.ainvoke({
                "input": row["question"],
                "context": row["contexts"]
            })
            outputs.append(answer)

        logger.debug("Got all the answers")
        df["answers"] = [output["answer"] for output in outputs]

        df.to_csv((Path.cwd() / "eval" / "data" / "wikiqa_our_answers.csv"),
                  index=False)

    dataset = Dataset.from_pandas(df)
    config = RunConfig(
        max_retries=10,
        max_wait=60,
        thread_timeout=300,
        log_tenacity=True
    )

    # if faithfullness --> None and `Runner in Executor raised an exception` in logs:
    # https://github.com/explodinggradients/ragas/issues/1099
    # 1) check _create_statements_prompt and _create_nli_prompt methods
    # 2) check parsers _statements_output_parser, _faithfulness_output_parser
    # 3) verify prompts

    scores = evaluate(
        llm=langchain_llm,
        dataset=dataset,
        metrics=metrics,
        embeddings=langchain_embeddings,
        raise_exceptions=True,
        run_config=config
    )

    # chunk_size = 2
    # start = 0
    # chunks = [df.iloc[start:chunk_size] for start in range(0, df.shape[0], chunk_size)]
    #
    # with ThreadPoolExecutor() as executor:
    #     # Submit all chunks to the executor
    #     futures = {executor.submit(process_fn, chunk): chunk for chunk in chunks}
    #
    #     # As each future completes, store the result
    #     for future in as_completed(futures):
    #         result = future.result()
    #         scores.append(result)
    # pprint(scores)

    return scores


async def eval_rag(
        simulacra: Simulacra,
        client: Optional[Client] = None,
) -> ExperimentResults:
    correctness_evaluator = get_correctness_evaluator(llm=simulacra.llm)
    relevancy_evaluator = get_relevancy_evaluator(llm=simulacra.llm)
    target = simulacra.answer

    results = await aevaluate(
        target,
        data="wikiQA_abmax_mvp",
        evaluators=[
            correctness_evaluator,
            relevancy_evaluator
        ],
        client=client,
    )

    return results


async def main():
    load_dotenv()
    llama = OllamaLLM(model="llama3.1")
    df = pd.read_csv(Path(os.getcwd(), "eval", "data", "wikiqa.csv"))
    df.rename(columns={"answers": "answer"}, inplace=True)

    data = df[[
        "answer",
        "question",
        "contexts",
        "ground_truth"
    ]]

    # contexts = data[["contexts"]]
    # for _, el in contexts.iterrows():
    #     print(type(el[0]), pformat(el[0]))
    #     exit(0)
    # logger.debug(df.columns)
    # data = Dataset.from_pandas(df)
    # docs = DataFrameLoader.lazy_load(df[["contexts"]])
    data.contexts = data.contexts.apply(literal_eval)

    docs = [Document(d[0]) for d in data["contexts"]][:10]
    logger.info(f"num docs: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )

    splits = splitter.split_documents(docs)
    logger.info(f"num splits: {len(splits)}")

    # embedding_model = OllamaEmbeddings(model="llama3.1")
    # vectorstore = ElasticsearchStore(
    #     es_url=ELASTIC_URL,
    #     index_name="simulacra_memory",
    #     embedding=embedding_model,
    #     distance_strategy="COSINE",
    #     strategy=ElasticsearchStore.ApproxRetrievalStrategy(
    #         hybrid=True
    #     )
    # )

    # vectorstore = ElasticVectorSearch.from_documents(
    #     splits,
    #     embedding=embedding_model,
    #     elasticsearch_url=ELASTIC_URL,
    #     index_name="simulacra_memory"
    # )

    # retriever = vectorstore.as_retriever()
    retriever.add_documents(splits)

    eval_data_path = Path(os.getcwd(), "eval", "data")
    # results = {}
    #
    # with open(Path(eval_data_path, f"relevant_docs_found.json"), "w") as fp:
    #     json.dump(results, fp, indent=4)

    personal_path = Path(os.getcwd(), "init_module", "profiles_B5_approach", "person_0")
    simulacra = Simulacra(
        personal_path=personal_path,
        core_llm=llama,
    )

    eval_results = await eval_rag(
        target=simulacra.answer,
        # client=Client(),
        llm=simulacra.llm,
    )

    for result in eval_results:
        logger.info(f"{pformat(result)} ({type(result)})")

    # with open(Path(eval_data_path, f"relevant_docs_found.json"), "w") as fp:
    #     json.dump(eval_result, fp, indent=4)


if __name__ == '__main__':
    asyncio.run(main())
