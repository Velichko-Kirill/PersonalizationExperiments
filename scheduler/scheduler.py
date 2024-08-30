import asyncio
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from glob import glob

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama

from memory.retriever import setup_faiss_retriever, format_docs, generate_memories
from memory.prompts import retrieval_prompt
from scheduler.prompts import gen_action_prompt

from langchain_experimental.llms import ChatLlamaAPI
from llamaapi import LlamaAPI

from tqdm import tqdm

EMBEDDING_MODEL_ID = "nomic-embed-text"
# Must correspond EMBEDDING_MODEL_ID from init_module.person !


class Scheduler:
    def __init__(self, personal_path: str,
                 rag_chain: Union[RunnablePassthrough, RunnableSerializable[str, str]],
                 template_file: Union[Path, str],
                 llm: ChatOllama):
        """
        Инициализатор класса
        :param personal_path: Путь к директории с профилем и биографией агента
        :param template_file: Путь к файлу с расписания агента (состоит из 7 строк для парсинга)
        :param llm: LLM модель для выполнения промптов
        """

        self.personal_path = personal_path
        self.template_file = template_file
        self.rag_chain = rag_chain

        self.schedule = None
        self.gen_action_chain = gen_action_prompt | llm

    def parse_day_description(self, day_string: str) -> tuple[tuple[str, int], ...]:
        """
        Метод для парсинга строки с расписанием дня

        Пример работы:
        "r2p13t1" -> (("recreation", 2), ("professional", 13), ("transit", 1))

        :param day_string: Строка из цифр и символов {r,t,s,p}
        :return: Кортеж кортежей вида ("тип_действия", кол-во часов)
        """

        result = []
        hours_sum = 0
        i = 0

        while i < len(day_string):
            match day_string[i]:
                case "r":
                    category = 'recreation'
                case "t":
                    category = 'transit'
                case "s":
                    category = 'social'
                case "p":
                    category = 'professional'
                case "\n":
                    break
                case _:
                    raise ValueError(f"Unexpected character: {day_string[i]} at pos {i} in {day_string}")
            i += 1

            num_str = ''
            while i < len(day_string) and day_string[i].isdigit():
                num_str += day_string[i]
                i += 1
            num = int(num_str)

            result.append((category, num))
            hours_sum += num

        # Ограничение на 16 часов, т.к. с 12 до 8 - сон
        if hours_sum > 16:
            raise ValueError(f"day description cant be longer than 16 hours (found {hours_sum}): {day_string}")

        result.insert(0, ('sleep', 8))
        result.append(('sleep', 16 - hours_sum))

        return tuple(result)

    def init_schedule(self, start_date: str):
        """
        Метод парсит строки из текущего шаблона расписания агента и инициализирует пустое расписание.

        :param start_date: Строка с датой начала недели в формате "dd/mm/YY"
        """

        schedule = []
        with open(self.template_file, 'r') as file:
            for line in file:
                if line[0] == "#" or line == "\n":
                    continue
                schedule.append(self.parse_day_description(line))

        activity_col = [activity for day in schedule for activity, count in day for _ in range(count)]
        start_datetime = pd.to_datetime(start_date, format="%d/%m/%y")
        time_index = pd.date_range(start=start_datetime, periods=len(activity_col), freq='h')

        schedule_df = pd.DataFrame({'timestamp': time_index,
                                    'weekday': time_index.day_name(),
                                    'activity_type': activity_col})
        self.schedule = schedule_df

    async def fill_schedule(self, verbose: Optional[bool] = True):
        """
        Метод заполняет текущее расписание конкретными действиями агента

        :param start_date: Строка с датой начала недели в формате "dd/mm/YY"
        :param verbose: Контроль вывода при работе процедуры
        """
        if not isinstance(self.schedule, pd.DataFrame):
            raise TypeError(f"Wrong schedule type: expected pd.DataFrame, "
                            f"but found {type(self.schedule)}. Is it initialized?")

        with open(self.personal_path + 'brief_biography.txt', 'r') as file:
            bio = file.read()

        with open(self.personal_path + 'profile.json', 'r') as file:
            profile = file.read()

        for i, row in tqdm(self.schedule.iterrows(), total=self.schedule.shape[0],
                           desc="Filling up the schedule", disable=not verbose):

            action_type = row["activity_type"]
            match action_type:
                case "sleep":
                    self.schedule.at[i, 'taken_action'] = "Sleeping"
                    continue
                case "professional":
                    action_type = "must do something professional or work related."
                case "transit":
                    action_type = "are in transit to your next destination."
                    action_type += "Transit activities refer to the act of traveling from one place to another. " \
                                   "So you have to be moving somewhere and possible do something while in transit"
                case "recreation":
                    action_type = "must do something recreational."
                case "social":
                    action_type = "must socialize."

            timestamp = row["timestamp"].strftime("%H:%M")
            prev_action = self.schedule.at[i - 1, "taken_action"]

            if prev_action == "Sleeping":
                question = f"What activities can you do in the morning?"
            else:
                question = f"What activities, similar to \"{prev_action}\" can agent do?"

            memories = rag_chain.invoke(input=question)

            self.schedule.at[i, 'memories'] = memories

            model_ans = self.gen_action_chain.invoke({"bio": bio,
                                                      "profile": profile,
                                                      "timestamp": timestamp,
                                                      "weekday": row["weekday"],
                                                      "action_type": action_type,
                                                      "prev_action": prev_action,
                                                      "memories": memories}).content
            self.schedule.at[i, 'taken_action'] = model_ans

    def change_schedule_template(self, template_file: str):
        """
        Меняет расписание агента
        :param template_file: Путь к файлу с расписания агента (состоит из 7 строк для парсинга)
        """
        self.template_file = template_file

    def change_personal_path(self, personal_path: str):
        """
        Меняет профиль и биографию агента
        :param personal_path: Путь к директории с профилем и биографией агента
        """
        self.personal_path = personal_path

    def schedule_to_csv(self, output_file: str):
        """
        Сохраняет текущее расписание агента в формате .csv
        :param output_file: Путь для сохранения расписания
        """
        if isinstance(self.schedule, pd.DataFrame):
            self.schedule.to_csv(output_file, index=False)
        else:
            raise TypeError(f"Wrong schedule type: expected pd.DataFrame, "
                            f"but found {type(self.schedule)}. Is it initialized?")


if __name__ == "__main__":
    start_date = "05/08/24"
    used_profiles = ["roman_alexeev", "kaitlin_soto", "jessica_blankenship"]

    personal_path_map = {"roman_alexeev": "../bio_gen/results/modified/person_0/",
                         "jessica_blankenship": "../bio_gen/results/modified/person_7/",
                         "kaitlin_soto": "../bio_gen/results/modified/person_27/"}

    llama_client = LlamaAPI(os.getenv("LLAMA_API_KEY"))
    # llama = ChatLlamaAPI(client=llama_client)
    llama = ChatOllama(model="llama3.1")

    for input_file in glob("schedule_profiles/*.txt"):
        base_name = os.path.basename(input_file).split(".")[0]
        if base_name not in used_profiles:
            continue
        print(base_name)

        retriever: VectorStoreRetriever = setup_faiss_retriever(
            memories=generate_memories(personal_path_map[base_name], llm=llama),
            num_relevant_docs=4,
            embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL_ID),
        )
        rag_chain = (
                {
                    "context": retriever | format_docs,
                    "input": RunnablePassthrough(),
                }
                | retrieval_prompt
                | llama
                | StrOutputParser()
        )
        scheduler = Scheduler(personal_path=personal_path_map[base_name],
                              rag_chain=rag_chain,
                              template_file=input_file,
                              llm=llama)

        scheduler.init_schedule(start_date)
        asyncio.run(scheduler.fill_schedule())
        scheduler.schedule_to_csv(f"result/{base_name}_schedule_filled_mem.csv")
