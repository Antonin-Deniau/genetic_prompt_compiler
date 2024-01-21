import random
from typing import Callable
from dataclasses import dataclass


@dataclass
class RuleBasedFitnessConfig:
    fitness_llm: Callable[[str], str]
    student: Callable[[str, str], str]
    rules: list[str]
    rating_notation: int
    train_examples: list[str]
    example_amount: int


FITNESS_PROMPT = """
Rate the quality of the generated answer, given a sentence, on a scale from 1 to {x}, 1 being the worst and {x} being the best.
The answer should respect the following rules:
{rules}

The answer is:
`{answer}`

Your rating / {x}:
"""


def _evaluate_answer(args: RuleBasedFitnessConfig, answer: str) -> float | None:
    rules = "\n".join([f"\t- {rule}" for rule in args.rules])
    content = FITNESS_PROMPT.format(rules=rules, answer=answer, x=args.rating_notation)

    res = args.fitness_llm(content)

    try:
        return float(res.strip())
    except:
        return 0


def rule_based_fitness(args: RuleBasedFitnessConfig, prompt: str) -> float:
    examples = [random.choice(args.train_examples) for _ in range(args.example_amount)]

    fitnesses = []

    for q in examples:
        a = args.student(prompt, q)
        f = _evaluate_answer(args, a)

        fitnesses.append(f)

    return sum(fitnesses) / len(fitnesses)
