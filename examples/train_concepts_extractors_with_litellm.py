import logging
from litellm import completion

import genetic_prompt_compiler

from genetic_prompt_compiler import GeneticCompilerArgs
from genetic_prompt_compiler.mutate import rule_based_mutate, RuleBasedMutateConfig
from genetic_prompt_compiler.ranking import top_n_ranking, TopNRankingConfig
from genetic_prompt_compiler.fitness import rule_based_fitness, RuleBasedFitnessConfig

logging.basicConfig(level=logging.INFO)

"""
This example shows how to use the genetic_prompt_compiler to train a concept extractor

It use litellm to call openai's API for the fitness and mutation functions
And it is used to train a concept extractor on VLLM, via litellm
"""

initial_prompt = (
    "Extract ideas from the user's question, and format them into a JSON list"
)

rules = [
    "It must be a list of concepts",
    "Each concepts must be a json string",
    "Each concepts must be present in the sentence",
    "The concepts should be strings only, not objects",
    "The person speaking must be present in the concepts",
    "It must be in english",
    "No comments added after the list",
    "No new sentence generated",
    "The JSON must be valid",
    'It should look like this: `["concept1", "concept2", "concept3"]`',
]

test_data = [
    "Could you please pass me the salt?",
    "Did you remember to pick up the groceries after work?",
    "Remember to turn off the lights before you leave the room.",
    "The traffic is terrible today, should we try taking the train?",
    "Can we go see the new superhero movie this weekend?",
    "Don't forget to wash your hands before dinner.",
    "Can you believe the final score of the game last night?",
    "Did you remember to lock the car before we left?",
]


def fitness_llm(q: str) -> str:
    messages = [
        {
            "role": "user",
            "content": q,
        },
    ]

    res = completion(
        messages=messages,
        model="gpt-4",
        temperature=0.1,
    )

    return res.choices[0].message.content


def student_llm(prompt: str, q: str) -> str:
    formated_prompt = f"{prompt}\n\nI need to extract the concepts from:\n\t`{q}`"
    messages = [
        {
            "role": "user",
            "content": formated_prompt,
        },
    ]

    res = completion(
        messages=messages,
        model="openai/TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        api_base="http://127.0.0.1:8000/v1",
        temperature=0.1,
    )

    return res.choices[0].message.content


def mutation_llm(q: str) -> str:
    messages = [
        {
            "role": "user",
            "content": q,
        },
    ]

    res = completion(
        messages=messages,
        model="gpt-4",
        temperature=0.3,
    )

    return res.choices[0].message.content


args = GeneticCompilerArgs(
    mutate=rule_based_mutate,
    ranking=top_n_ranking,
    fitness=rule_based_fitness,
    ranking_config=TopNRankingConfig(
        top_n=5,
    ),
    mutation_config=RuleBasedMutateConfig(
        mutation_llm=mutation_llm,
        rules=rules,
    ),
    fitnes_config=RuleBasedFitnessConfig(
        fitness_llm=fitness_llm,
        student=student_llm,
        rules=rules,
        rating_notation=10,
        train_examples=test_data,
        example_amount=3,
    ),
    popultation_size=10,
    iterations=5,
    initial_prompts=[initial_prompt],
)


for population in genetic_prompt_compiler.run(args):
    print(f"Top prompts:")
    for prompt in population:
        print(f"\t - {prompt}")
