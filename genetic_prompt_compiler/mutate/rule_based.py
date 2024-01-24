import random
from typing import Callable, Any
from dataclasses import dataclass, field


@dataclass
class Technique:
    prompt: str
    presence: float


DEFAULT_TECHNIQUES = [
    Technique(
        prompt="Use the expert technique `You are an expert in {topic}`",
        presence=0.3,
    ),
    Technique(
        prompt="Use the Chain of Thought technique `Let's think step by step...`",
        presence=0.3,
    ),
    Technique(
        prompt="Use some examples `Here are some examples of answers: {examples}`",
        presence=0.3,
    ),
]


@dataclass
class RuleBasedMutateConfig:
    mutation_llm: Callable[[str], str]
    rules: list[str]
    techniques: list[Technique] = field(default_factory=lambda: DEFAULT_TECHNIQUES)


MUTATE_PROMPT = """
Generate a new prompt that take the best of each prompts, following the following rules:
{rules}

For the prompt(s):
{prompts}

You can use some known techniques to generate new prompts, like:
{techniques}

The new prompt is:
"""


def rule_based_mutate(args: RuleBasedMutateConfig, population: list[str]):
    used_techniques = []
    for technique in args.techniques:
        if random.random() < technique.presence:
            used_techniques.append(technique.prompt)

    content = MUTATE_PROMPT.format(
        prompts="\n".join([f"\t- {prompt}" for prompt in population]),
        rules=args.rules,
        techniques="\n".join([f"\t- {technique}" for technique in used_techniques]),
    )

    return args.mutation_llm(content)
