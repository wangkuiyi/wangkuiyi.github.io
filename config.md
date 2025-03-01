# The Right Way to Configure LLM Training

When reading the title, many people might immediately think of configuration files in JSON or YAML, or configuration libraries like Hydra. However, these approaches fall short in industrial AI systems. Let’s explore why that is and examine better solutions by looking at AXLearn, an open-source framework used to train the foundation model behind Apple Intelligence.

## Why Traditional Configuration Methods Fall Short

It’s common to configure a program by providing command-line options and/or configuration files. A typical programming practice is to pass configurations as arguments to function calls and class instantiations.

When the program in question is a training pipeline, the configuration defines a training recipe. For example:

- A wide-and-deep ranking model in industry might have over 45,000 configuration fields, most of which specify which features to use.
- An LLM pre-training recipe often has a similar number of fields.

For researchers, managing and discussing such a massive configuration is nearly impossible—staring at a 45,000-line JSON file is intractable. It becomes even worse when trying to trace which configuration fields correspond to which function calls and class instantiations.

Command-line options libraries (such as absl-py) and parsers (like Hydra for YAML/JSON) do not help in mapping configuration fields back to function arguments.

## A Smarter Approach: AXLearn, TensorFlow Lingvo, and Google Fiddle

Instead of parsing a static configuration and using it as arguments, AXLearn and Fiddle take the opposite approach:

- They convert function/class parameters into configurations rather than the other way around.
- This method preserves structure, enables introspection, and simplifies experiment management.

Let’s see how this approach works.

------------------------------------------------------------

## Converting Parameters to Configurations

Consider a simple class:

```python
class Optimizer:
    def __init__(self, learning_rate: float, momentum: float = 0.9):
        ...
```

### Using Google Fiddle

Fiddle creates a configuration class dynamically, with data members corresponding to the class constructor parameters. You can then instantiate the class using the configuration instance:

```python
import fiddle

# Define the configuration.
optimizer_config = fiddle.Config(Optimizer, learning_rate=0.1)

# Instantiate the Optimizer using the configuration.
optimizer_instance = fiddle.build(optimizer_config)
```

### Using AXLearn

AXLearn provides a similar API:

```python
from axlearn.common import config as ax_config

# Define the configuration.
optimizer_config = ax_config.config_for_class(Optimizer).set(learning_rate=0.1)

# Instantiate the Optimizer using the configuration.
optimizer_instance = optimizer_config.instantiate()
```

This approach ensures that configurations remain structured and traceable.

## Nesting Configurations

Many training programs involve multiple components, such as a dataset, model, optimizer, and trainer. Consider the following Trainer class:

```python
class Trainer:
    def __init__(self, dataset: Dataset, model: Model, optimizer: Optimizer, num_steps: int):
        ...
```

### Using Google Fiddle

Fiddle allows for nested configurations, making it easy to configure Trainer along with its dependencies (Dataset, Model, and Optimizer):

```python
trainer_config = fiddle.Config(
    Trainer,
    dataset=fiddle.Config(Dataset, ...),
    model=fiddle.Config(Model, ...),
    optimizer=optimizer_config,  # Reusing the optimizer config defined earlier.
    num_steps=10000,
)

trainer_instance = fiddle.build(trainer_config)
```

### Using AXLearn

Similarly, AXLearn enables nesting configurations in a structured way:

```python
trainer_config = ax_config.config_for_class(Trainer).set(
    dataset=ax_config.config_for_class(Dataset).set(...),
    model=ax_config.config_for_class(Model).set(...),
    optimizer=optimizer_config,  # Reusing the optimizer config defined earlier.
    num_steps=10000,
)

trainer_instance = trainer_config.instantiate()
```

------------------------------------------------------------

## Why This Matters

By converting parameters into structured configurations, AXLearn and Fiddle provide:

- Better traceability – Easily map configurations to function calls.
- Improved maintainability – Avoid managing huge static configuration files.
- Efficient code review – Focus on changes (diffs) rather than massive JSON/YAML files.
- Seamless integration – Easily nest configurations for complex ML pipelines.

This configuration-first approach is already powering AI systems at scale, including Apple Intelligence. If you are working on LLM training, it’s worth considering a shift away from traditional config files to a structured, introspective configuration system like AXLearn.
