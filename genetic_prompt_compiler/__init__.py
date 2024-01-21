from functools import partial
import logging
from typing import Callable, TypeVar, Generic
from dataclasses import dataclass

from tqdm import tqdm

M = TypeVar("M")
F = TypeVar("F")
R = TypeVar("R")


@dataclass
class GeneticCompilerArgs(Generic[M, F, R]):
    mutate: Callable[[M, list[str]], str]
    ranking: Callable[[R, list[tuple[str, float]]], list[str]]
    fitness: Callable[[F, str], float | None]

    ranking_config: R
    mutation_config: M
    fitnes_config: F

    popultation_size: int
    iterations: int
    initial_prompts: list[str]

def run(args: GeneticCompilerArgs):
    mutate = partial(args.mutate, args.mutation_config)
    ranking = partial(args.ranking, args.ranking_config)
    fitness = partial(args.fitness, args.fitnes_config)

    # Bootstrap
    population = args.initial_prompts

    initial_prompts_len = len(args.initial_prompts)
    if initial_prompts_len > args.popultation_size:
        raise ValueError(
            f"Initial prompts length ({initial_prompts_len}) is greater than population size ({args.popultation_size})"
        )

    for i in tqdm(
        range(args.popultation_size - initial_prompts_len), desc="Bootstrapping prompts"
    ):
        population.append(mutate(args.initial_prompts))
        logging.info(
            f"{i+1}/{args.popultation_size-initial_prompts_len} prompts generated"
        )

    # Evolution
    for generation in tqdm(range(args.iterations), desc="Evolving prompts"):
        # Evaluate fitness
        evaluated_population = []

        logging.info(f"Generation {generation}")
        for i, prompt in enumerate(population):
            logging.info(f"Evaluating prompt {i}/{args.popultation_size}")
            fit = fitness(prompt)
            evaluated_population.append((prompt, fit))

        # Ranking
        top_n_population = ranking(evaluated_population)
        yield top_n_population

        # Mutation
        new_population = []
        for _ in range(args.popultation_size - args.ranking_config.top_n):
            new_population.append(mutate(top_n_population))

        new_population += top_n_population

        population = new_population
