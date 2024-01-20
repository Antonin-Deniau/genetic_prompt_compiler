import random
import logging
from typing import Any
from dataclasses import dataclass

from tqdm import tqdm
from litellm import completion


@dataclass
class GeneticCompilerArgs:
    example_amount: int
    top_n: int
    popultation_size: int
    iterations: int
    initial_prompt: str
    rules: list[str]
    log_level: str
    train_data: list[tuple[str, str]]
    fitness_model_args: dict[str, Any]
    student_model_args: dict[str, Any]
    mutation_model_args: dict[str, Any]
    rating_notation: int


FITNESS_PROMPT = """
Rate the quality of the generated answer, given a sentence, on a scale from 1 to {x}, 1 being the worst and {x} being the best.
The answer should respect the following rules:
{rules}

The answer is:
`{answer}`

Your rating / {x}:
"""

MUTATE_PROMPT = """
Generate a new prompt that take the best of each prompts, following the following rules:
{rules}

For the prompt(s):
    - `{prompts}`

The new prompt is:
"""


def student(args, prompt, q):
    messages = [
        {
            "role": "user",
            "content": "{}\n\nExtract the concepts of the sentence: \n\n - `{}`".format(
                prompt, q
            ),
        },
    ]

    res = completion(
        messages=messages,
        **args.student_model_args,
    )

    return res.choices[0].message.content


def fitness(args: GeneticCompilerArgs, answer: str):
    rules = "\n".join([f"\t- {rule}" for rule in args.rules])
    messages = [
        {
            "role": "user",
            "content": FITNESS_PROMPT.format(rules=rules, answer=answer, x=args.rating_notation),
        },
    ]

    res = completion(
        messages=messages,
        **args.fitness_model_args,
    )

    try:
        return float(res.choices[0].message.content.strip())
    except:
        return 0


def mutate(args: GeneticCompilerArgs, population: list[str]):
    messages = [
        {
            "role": "user",
            "content": MUTATE_PROMPT.format(
                prompts="\n".join(population),
                rules=args.rules,
            ),
        },
    ]

    res = completion(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
    )

    return res.choices[0].message.content


def evaluate_prompt(args: GeneticCompilerArgs, prompt: str) -> float:
    examples = [random.choice(args.train_data) for _ in range(args.example_amount)]

    fitnesses = []

    for q, _ in examples:
        a = student(args, prompt, q)
        f = fitness(args, a)

        fitnesses.append(f)

    return sum(fitnesses) / len(fitnesses)


def run(args: GeneticCompilerArgs):
    logging.basicConfig(level=args.log_level)

    # Bootstrap
    population = [args.initial_prompt]

    for i in tqdm(range(args.popultation_size - 1), desc="Bootstrapping prompts"):
        population.append(mutate(args, [args.initial_prompt]))
        logging.info(f"{i+1}/{args.popultation_size-1} prompts generated")

    # Evolution
    for generation in tqdm(range(args.iterations), desc="Evolving prompts"):
        # Evaluate
        evaluated_population = []

        logging.info(f"Generation {generation}")
        for i, prompt in enumerate(population):
            logging.info(f"Evaluating prompt {i}/{args.popultation_size}")
            fit = evaluate_prompt(args, prompt)
            evaluated_population.append((prompt, fit))

        # Select
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Top {args.top_n} prompts:")
        for prompt, fit in evaluated_population[: args.top_n]:
            logging.info(f"{fit}\t{prompt}")

        logging.info(f"Generation new population...")
        new_population = []

        top_n_population = [x[0] for x in evaluated_population[: args.top_n]]

        for _ in range(args.popultation_size - args.top_n):
            new_population.append(mutate(args, top_n_population))

        new_population += top_n_population

        population = new_population

        yield population
