
<b> 0. Требования </b>
--
2.90 GHz, 3260316 Kib RAM, 300 Mb Drive + ~8.5Гб для текущих моделей

python 3.12, docker, poetry

<b> 1. Установка </b>
--

1) Запустите контейнер с <a href=https://ollama.com/>Ollama</a> командой

`docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`

(режим CPU-only)

2) Загрузите модели

`ollama pull llama3.1`

`ollama pull nomic-embed-text`

3) Активируйте виртуальную среду poetry и установите все необходимые зависимости

 `poetry shell && poetry install`

4) (Опционально) Настройте переменные окружения (например, создав файл .env)

- Для использования ElasticSearch engine:

ELASTIC_PORT = '9200'

MODEL_ID='elastic__distilbert-base-uncased-finetuned-conll03-english'

- Для эвалюации RAG посредством `langsmith`:

 - LANGCHAIN_API_KEY=\<your-api-key\>
 - LANGCHAIN_TRACING_V2='true'
 - LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
 - LANGSMITH_TRACING=true



<b> 2. Модули </b>
--

<b> ВАЖНО! </b>
Запуская модули из командной строки, проверьте, что используете python из актуального окружения poetry

<b> 2.1 Инициализация </b>

Для использования персонажа необходимо предварительно создать его личную директорию со всеми исходными данными. По умолчанию такие директории располагаются в init_module/persons. В данный момент поддерживается 2 подхода: Big Five и MBTI, отличающиеся только наполнением полей `personal_traits`. В директории person_\<person_id\> (обязательный формат названия),
должен быть создан файл profile.json .
Формат файла следующий:

<i> Обязательные </i> поля:
  - 'id'
  - 'name'
  - 'birthday'

<i>  желательные: </i>

  - 'unique_quality'
  - 'education'
  - 'career'
  - 'family'
  - 'residence'

Соответствующая pydantic модель находится в `init_module/person.py` (class Profile), пример - `init_module/example_profile.json `

Остальные файлы, необходимые для инициализации, будут сохраняться в той же директории, так что <i> проверьте, что есть права на запись туда. </i>

Запуск осуществляется из корневой директории командой

`python -m bio_gen -p [OPTIONS]`

options:
- --personal-path -p : путь до директории с файлом `profile.json`

Список итоговых файлов в `personal_path`:
  - enriched_biography.txt
  - brief_biography.txt
  - chunks.csv
  - profile.json

<b> 2.2 Действия </b>

Основной модуль, реализующий действия персоны.
Функционал:

  - инициализирует персону. Для этого используются файлы, созданные в результате работы модуля 1.

  - Читает текущие наблюдения из среды. По умолчанию они находятся в файле main/observations.txt и преставляют собой текстовые фреймы, разделённые символами "###".

  - Генерирует реакции персоны

  - Сохраняет результат в виде словаря {"observation": reaction} в `<personal_path>/reactions.json`

Запуск осуществляется из корневой директории командой

`python -m main -p [OPTIONS]`

options:

  - "--personal_path", "-p"
  - "--core-model", "-llm"
  - "--observations-path", "-o"


<b> 2.3 Память </b>

В данный момент основная функция для retriever'а - memory.retriever.setup_faiss_retriever
Именно она вызывается в методе __init\__ у init_module.person.Simulacra
Для перехода на эластик можно использовать
memory.retriever.SimulacraMemoryStorage
По умолчанию память инициализируется у Simulacra,
обращаться к содержанию можно через `simulacra.retriever` и `simulacra.rag_chain`



<b> 2.4 Планирование </b>

Базовый класс: scheduler.scheduler.Scheduler

Вызывается при инициализации init_module.person.Simulacra,

обращаться можно через `simulacra.scheduler`:

Инициализация планирования: `scheduler.init_schedule(start_date)`

Заполнение расписания:
`asyncio.run(scheduler.fill_schedule())` или `await scheduler.fill_schedule()`

Сохранение расписания:
`scheduler.schedule_to_csv(<path/to/csv>)`

<b> 2.5 Рефлексия </b>

TODO

https://blog.langchain.dev/reflection-agents/

<b> 2.6 Мировоззрение </b>

TODO

<здесь про психологические тесты и вот это вот всё>

 модуль worldview
