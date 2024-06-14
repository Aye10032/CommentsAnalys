import os
from enum import IntEnum
from typing import List, Tuple

import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from pandas import DataFrame
from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

import pandas as pd

from Prompt import *


class GenerateAnswer(BaseModel):
    """返回的观点候选列表，如果没有生成新的观点，则**直接返回**用户给出的原列表；否则，返回添加了新观点的列表"""
    lists: List[str] = Field(
        ...,
        description='候选观点列表。如果没有生成新的观点，则直接返回用户给出的**原列表**；否则，返回添加了新观点的列表'
    )


class ChooseAnswer(BaseModel):
    """包含两个值。内容分别是根据给出的评论，从观点候选列表选择的与之最匹配的观点和该观点对应的数组索引"""
    answer: str = Field(
        ...,
        description='选择的观点文本'
    )

    index: int = Field(
        ...,
        description='选择的观点对应对数组索引'
    )


class MissionType(IntEnum):
    UPDATE = 0
    CHOOSE = 1


def load_llm() -> ChatOpenAI:
    with open('api_key.yaml', 'r') as yaml_f:
        yaml_text = yaml_f.read()
    llm_api = yaml.load(yaml_text, Loader=yaml.FullLoader)['zhipu']

    llm = ChatOpenAI(
        model="glm-4",
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
        openai_api_key=llm_api,
        temperature=0,
    )

    return llm


def answer(lists: list, comment: str, mission_type: MissionType) -> dict:
    if mission_type == MissionType.UPDATE:
        parser = JsonOutputParser(pydantic_object=GenerateAnswer)
    else:
        parser = JsonOutputParser(pydantic_object=ChooseAnswer)

    system_prompt = PromptTemplate(
        template=SYSTEM,
        input_variables=["format_instructions"],
    )

    system_str = system_prompt.format(
        format_instructions=parser.get_format_instructions(),
    )

    if mission_type == MissionType.UPDATE:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_str),
            ('human', ASK_UPDATE)
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_str),
            ('human', ASK_CHOOSE)
        ])

    llm = load_llm()

    chain = prompt | llm | parser

    result = chain.invoke({'comment': comment, 'lists': lists})

    return result


def load_data(data_path: str, output_path: str, init: bool = False) -> Tuple[DataFrame, list]:
    if init:
        with open(data_path, 'r', encoding='utf-8') as f:
            comments = f.readlines()

        comment_df = pd.DataFrame([{'comment': comment.replace('\n', '')} for comment in comments])
        comment_df['generate'] = False
        comment_df['choose'] = False

        opi_list = []
    else:
        comment_df = pd.read_csv(os.path.join(output_path, 'news.csv'), encoding='utf-8')

        with open(os.path.join(output_path, 'ops.txt'), 'r', encoding='utf-8') as op_f:
            opi_list = op_f.readlines()
            opi_list = [opi.replace('\n', '') for opi in opi_list]

    return comment_df, opi_list


def run(data: DataFrame, lists: list, output_path: str, init: bool = False) -> None:
    opinion_file = os.path.join(output_path, 'ops.txt')
    record_file = os.path.join(output_path, 'news.csv')
    result_file = os.path.join(output_path, 'result.csv')

    for index, row in tqdm(data.iterrows(), total=30):
        if row['generate']:
            continue

        comment = row['comment']
        lists.extend(answer(lists, comment, MissionType.UPDATE)['lists'])
        lists = list(set(lists))
        with open(opinion_file, 'w', encoding='utf-8') as op_f:
            op_f.write('\n'.join(lists))
        data.at[index, 'generate'] = True
        data.to_csv(record_file, encoding='utf-8', index=False)

    if init or not os.path.exists(result_file):
        result = {key: 0 for key in lists}
        pd_list = [{'opi': key, 'count': value} for key, value in result.items()]
        df = pd.DataFrame(pd_list)
        df.to_csv(result_file, encoding='utf-8', index=False)
    else:
        df = pd.read_csv(result_file, encoding='utf-8')

    for index, row in tqdm(data.iterrows(), total=30):
        if row['choose']:
            continue

        comment = row['comment']
        ans = answer(lists, comment, MissionType.CHOOSE)
        df.at[ans['index'], 'count'] += 1
        df.to_csv(result_file, encoding='utf-8', index=False)

        data.at[index, 'choose'] = True
        data.to_csv(record_file, encoding='utf-8', index=False)


def main() -> None:
    init = False
    data_file = 'data/news.txt'
    output_path = 'output'

    data, lists = load_data(data_file, output_path, init)
    # 截取前十条数据
    data = data[:30]

    os.makedirs(output_path, exist_ok=True)

    run(data, lists, output_path, init)


if __name__ == '__main__':
    main()
