from typing import List

SEP_TOKEN: str = "[SEP]"
ENCODER_ID = "Muennighoff/SGPT-125M-weightedmean-nli-bitfit"
NUM_SAMPLES: int = 20000

RUS_CITIES: List[str] = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Челябинск",
    "Красноярск"
]

SUMMARIZER_ID: str = "IlyaGusev/mbart_ru_sum_gazeta"

# number of attemts for parsing Llama's outputs by langchain utils
NUM_ATTEMPTS: int = 10