from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate


def build_summarize_chain(template, model) -> Chain:
    summarize_prompt = PromptTemplate(
        template=template,
        input_variables=["text", "num_words"])

    return summarize_prompt | model


def build_psycho_desc_chain(template, model) -> Chain:
    psycho_desc_prompt = PromptTemplate(
        template=template,
        input_variables=["psycho_profile"]
    )

    return psycho_desc_prompt | model


def build_primary_chain(template, model, parser=None) -> Chain:
    core_prompt = PromptTemplate(
        template=template,
        input_variables=["traits"],
    )

    if parser is not None:
        return core_prompt | model | parser
    return core_prompt | model


def build_fact_extraction_chain(template, model, parser) -> Chain:
    extraction_prompt = PromptTemplate.from_template(
        template=template
    )

    return extraction_prompt | model | parser


def build_enrichment_chain(template, model, parser=None) -> Chain:
    expanding_prompt = PromptTemplate(
        template=template,
        input_variables=[
            "chunk",
            "importance",
            "redundancy",
            "psychological",
        ],
    )

    if parser is not None:
        return expanding_prompt | model | parser

    return expanding_prompt | model


def build_roleplay_chain(template, model, parser=None) -> Chain:
    roleplay_prompt = PromptTemplate(
        template=template,
        input_variables=["description", "situation"]
    )

    if parser is not None:
        return roleplay_prompt | model | parser

    return roleplay_prompt | model
