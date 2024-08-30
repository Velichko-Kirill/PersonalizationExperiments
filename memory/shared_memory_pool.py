import asyncio
import logging
import os
import sys
import feedparser
from pprint import pprint, pformat

from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from bs4 import BeautifulSoup
from transformers import MBartTokenizer, MBartForConditionalGeneration

from memory.consts import (
    WEATHER_INDEX_MAPPING,
    REQUEST_EXAMPLE,
    NEWS_INDEX_MAPPING,
    LOCAL_NEWS_CHANNEL,
    SUMMARIZER_ID
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("DEBUG")


#################################################################################################
# Might be realized separately from GenerativeAgentMemory - the prototype of Shared Memory Pool #
#################################################################################################


class ElasticMemory:
    def __init__(self):
        load_dotenv()
        es_port = int(os.getenv('ELASTIC_PORT'))

        self.client = AsyncElasticsearch(
            verify_certs=False,
            hosts=[{
                'host': 'localhost',
                'port': es_port,
                'scheme': 'http'
            }], )

    @staticmethod
    async def far_to_celsius(fahrenheit: int, precision: int = 2) -> float:
        return round((fahrenheit - 32) * 5 / 9, precision)

    async def create_index(self, index, body):
        response = await self.client.indices.create(
            index=index,
            body=body,
            ignore=400
        )

        return response

    async def index_document(self, index, document):
        response = await self.client.index(index=index, document=document)

        return response

    async def search_documents(self, index, query):
        return await self.client.search(
            index=index,
            query=query,
            size=20)


def get_local_news():
    feed = feedparser.parse(LOCAL_NEWS_CHANNEL)
    entries_ = feed.entries
    print(len(entries_))
    print(feed.keys())

    for entry in entries_[:1]:
        for key in entry.keys():
            print(key, ":\n", pformat(entry[key]))

    return entries_


async def main():
    # initialize
    shared_memory = ElasticMemory()

    # Add weather
    await shared_memory.create_index("weather", WEATHER_INDEX_MAPPING)
    await shared_memory.client.index(
        index="weather",
        id=0,
        body=REQUEST_EXAMPLE
    )

    # Check
    _ = await shared_memory.search_documents(
        index="weather",
        query={
            "match": {
                "pressure": 30,
            }})

    # pprint(resp)

    # Add news
    await shared_memory.create_index("news", NEWS_INDEX_MAPPING)
    news = get_local_news()

    # Getting summarizer
    tokenizer = MBartTokenizer.from_pretrained(SUMMARIZER_ID)
    model = MBartForConditionalGeneration.from_pretrained(SUMMARIZER_ID)

    for id_, entry in enumerate(news):
        soup = BeautifulSoup(entry["summary"])
        article_text = soup.get_text()
        print(article_text)

        input_ids = tokenizer(
            [article_text],
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=4
        )[0]

        summary = tokenizer.decode(output_ids, skip_special_tokens=True)

        document = {
            "title": entry["title"],
            "published": entry["published"],
            "summary_detail": entry["summary_detail"],
            "sumary": summary,
        }

        print(f"summary:\n\n{summary} \n\n")
        await shared_memory.client.index(
            index="news",
            id=id_,
            document=document
        )

    await shared_memory.client.close()


if __name__ == '__main__':
    asyncio.run(main())
