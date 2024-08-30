import json
from pprint import pprint, pformat
from typing import Any, Dict, List

import requests
import feedparser

from memory.shared_memory_pool import get_local_news


def get_global_news() -> Dict[str, Any]:
    url = "https://newsdata.io/api/1/latest?country=ru&apikey=pub_4882469d59a2cc4794b2744ac648b01adedfa"
    resp = requests.get(url).json()

    pprint(resp)

    with open("../data/news_query.json", "w") as fp:
        json.dump(resp, fp=fp)

    return resp


def parse_entries(entries):
    queries_ = []
    for i, entry in enumerate(entries):
        queries_.append({
            "id": i,
            "title": entry.title,
            "summary": entry.summary,
        })

    return queries_


if __name__ == '__main__':
    entries = get_local_news()
    queries = parse_entries(entries)
    pprint(queries[0])

    with open("../data/local_news_query.json", "w") as fp:
        json.dump(queries, fp=fp)
