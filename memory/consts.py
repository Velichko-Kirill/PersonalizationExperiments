from typing import Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

WEATHER_INDEX_MAPPING: Dict[str, Any] = {
    "mappings": {
        "properties": {
            "timestamp": {"type": "date"},
            "location": {
                "type": "object",
                "properties": {
                    "city": {"type": "text"},
                    "country": {"type": "text"}
                }
            },
            "temperature": {
                "type": "object",
                "properties": {
                    "value": {"type": "float"},
                    "unit": {"type": "keyword"}
                }
            },
            "humidity": {"type": "integer"},
            "pressure": {"type": "integer"},
            "wind": {
                "type": "object",
                "properties": {
                    "speed": {"type": "float"},
                    "direction": {"type": "keyword"}
                }
            },
            "condition": {"type": "text"}
        }
    }
}

REQUEST_EXAMPLE: Dict[str, Any] = {
    "timestamp": '2024-07-16T10:00:00.000Z',
    "humidity": 57,
    "location": {
        "city": "Saint-Petersburg",
        "coordinates": {"lat": 59.9311, "lon": 30.3609}
    },
    "temperature": {"current": 20, "high": 25, "low": 18},
    "pressure": 30,
    "wind": {"speed": 10, "direction": "NW"},
    "condition": "Cloudy, with little precipitation"
}

NEWS_INDEX_MAPPING: Dict[str, Any] = {
    "mappings": {
        "title": {"type": "str"},
        "published": {"type": "date"},
        "summary": {"type": "str"},
        "summary_detail": {"type": "str"}
    }
}

VECTORSTORE_MAPPING: Dict[str, Dict[str, Dict[str, str]]] = {
    "properties": {
        "text": {"type": "text"},
        "vector": {
            "type": "dense_vector",
            "dims": 768
        }
    }
}

# Новости ВО
# LOCAL_NEWS_CHANNEL: str = "https://www.gov.spb.ru/gov/terr/reg_vasileostr/news/rss/"

LOCAL_NEWS_CHANNEL: str = "https://www.newswise.com/legacy/feed/channels.php?channel=6249"
SUMMARIZER_ID: str = "IlyaGusev/mbart_ru_sum_gazeta"
ELASTIC_URL: str = f"http://localhost:{os.getenv('ELASTIC_PORT')}"
