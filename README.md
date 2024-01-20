# Genetic Prompt Compiler

Optimize a prompt for a language model using a genetic algorithm.

```python
import genetic_compiler
from genetic_compiler import GeneticCompilerArgs

initial_prompt = ""

rules = [
    "Rule 1",
    "Rule 2",
    "Rule 3",
]

args = GeneticCompilerArgs(
    # The rating notation to use (X/10, X/5 etc.)
    rating_notation=10,
    # Amount of examples to test on prompts in each generation
    example_amount=3,
    # Top n prompts to keep in each generation
    top_n=3,
    # Amount of prompts to generate in each generation
    popultation_size=8,
    # Amount of generations to run
    iterations=3,
    # Initial prompt to start with.
    # This prompt will be kept for the first generation, alongside propulation_size - 1 mutated versions of it
    initial_prompt=initial_prompt,
    # Rules to test the prompts on
    rules=rules,
    # Log level
    log_level="INFO",
    # Test data to test the prompts on (Currently only the answers are used)
    train_data=train_data,
    # LiteLLM arguments for the fitness model (This llm will evaluate the prompts)
    fitness_model_args={
        "model": "gpt-4",
        "temperature": 0.1,
    },
    # LiteLLM arguments for the student model (This is the llm we want to optimize)
    student_model_args={
        "model": "openai/TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "api_base": "http://127.0.0.1:8000/v1",
        "temperature": 0.1,
    },
    # LiteLLM arguments for the mutation model (This llm will mutate the prompts)
    mutation_model_args={
        "model": "gpt-4",
        "temperature": 0.3,
    },
)

for population in genetic_compiler.run(args):
    top_prompt = population[0]
    print(f"Top prompt: {top_prompt}")

```
