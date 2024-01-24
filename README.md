[![PyPI version](https://badge.fury.io/py/genetic-prompt-compiler.svg)](https://badge.fury.io/py/genetic-prompt-compiler)
# Genetic Prompt Compiler

Optimize a prompt for a language model using a genetic algorithm.

# Installation
    
```bash
pip install genetic-prompt-compiler
```

# Usage

You can find complete examples in the [examples](examples) folder.

```python
import genetic_prompt_compiler
from genetic_prompt_compiler import GeneticCompilerArgs
from genetic_prompt_compiler.mutate import rule_based_mutate, RuleBasedMutateConfig, Technique
from genetic_prompt_compiler.ranking import top_n_ranking, TopNRankingConfig
from genetic_prompt_compiler.fitness import rule_based_fitness, RuleBasedFitnessConfig

initial_prompt = "Answer my question about the universe"

rules = [
    "It should be a good answer",
    "It should be factually correct",
    "It should be in english",
]

test_data [
    "Why is the sky blue?",
    "Who is the president of the United States?",
    "What is the capital of France?",
]

# The default techniques to use to mutate the prompts
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


args = GeneticCompilerArgs(
    # Mutation function to use
    mutate=rule_based_mutate,
    # Ranking function to use, will be used to select the prompts to keep in each generation
    ranking=top_n_ranking,
    # Fitness function to use, will be used to rank the prompts in each generation
    fitness=rule_based_fitness,
    # Ranking function arguments
    ranking_config=TopNRankingConfig(
        # Top n prompts to keep in each generation
        top_n=5,
    ),
    mutation_config=RuleBasedMutateConfig(
        # The llm function to use to mutate the prompts
        mutation_llm=lambda q: "",
        # Rules to generate the mutated prompts on
        rules=rules,
        # This is the default techniques used to mutate the prompts, you can omit this argument
        techniques=DEFAULT_TECHNIQUES,
    ),
    fitnes_config=RuleBasedFitnessConfig(
        # The llm function to use to rank the prompts
        fitness_llm=lambda q: "",
        # The llm function that you need to optimize
        student=lambda q: "",
        # Rules to test the prompts on
        rules=rules,
        # The rating notation to use (X/10, X/5 etc.)
        rating_notation=10,
        # Test data to test the prompts on
        train_examples=test_data,
        # Amount of examples to test on prompts in each generation
        example_amount=3,
    ),
    # Amount of prompts to generate in each generation
    popultation_size=10,
    # Amount of generations to run
    iterations=5,
    # Initial prompts to start with.
    # This prompts will be kept for the first generation, alongside propulation_size - len(initial_prompts) mutated versions of it
    initial_prompts=[initial_prompt],
)

for population in genetic_prompt_compiler.run(args):
    print(f"Top prompts:")
    for prompt in population:
        print(f"\t - {prompt}")
```