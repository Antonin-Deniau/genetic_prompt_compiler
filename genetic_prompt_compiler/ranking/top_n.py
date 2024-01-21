import logging
from dataclasses import dataclass


@dataclass
class TopNRankingConfig:
    top_n: int


def top_n_ranking(args: TopNRankingConfig, population: list[tuple[str, float]]):
    population.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"Top {args.top_n} prompts:")
    for prompt, fit in population[: args.top_n]:
        logging.info(f"{fit}\t{prompt}")

    logging.info(f"Generation new population...")

    return [x[0] for x in population[: args.top_n]]
