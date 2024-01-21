from typing import Callable
from dataclasses import dataclass


@dataclass
class RuleBasedMutateConfig:
    mutation_llm: Callable[[str], str]
    rules: list[str]


MUTATE_PROMPT = """
Generate a new prompt that take the best of each prompts, following the following rules:
{rules}

For the prompt(s):
    - `{prompts}`

The new prompt is:
"""


def rule_based_mutate(args: RuleBasedMutateConfig, population: list[str]):
    content = MUTATE_PROMPT.format(
        prompts="\n".join(population),
        rules=args.rules,
    )

    return args.mutation_llm(content)
