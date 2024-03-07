import argparse
from typing import List

from transformers import HfArgumentParser

from .base_args import BaseArguments


def init_parser(dataclass_types: List[BaseArguments]):
    return HfArgumentParser(dataclass_types)


def post_process_parsed(parsed):
    parsed_json = {}
    for p in parsed:
        parsed_json.update(p.__dict__)

    return parsed, parsed_json


def parse_json_file(
    dataclass_types: List[BaseArguments], json_file: str, allow_extra_keys=False
):
    parser = init_parser(dataclass_types)

    parsed = parser.parse_json_file(
        json_file=json_file,
        allow_extra_keys=allow_extra_keys,
    )

    return post_process_parsed(parsed)


def parse_yaml_file(
    dataclass_types: List[BaseArguments], yaml_file: str, allow_extra_keys=False
):
    parser = init_parser(dataclass_types)

    parsed = parser.parse_yaml_file(
        yaml_file=yaml_file,
        allow_extra_keys=allow_extra_keys,
    )

    return post_process_parsed(parsed)
