# Model Sharding in AXLearn

This note explains how [AXLearn](https://github.com/apple/axlearn), the open-source framework [used to train Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models), applies JAX's sharding primitives to enable distributed training across tens of thousands of accelerators. For JAX fundamentals on device meshes and `PartitionSpec`, see [mesh.md](mesh.md).

## Overview

Training large models requires distributing both data and model parameters across many devices. AXLearn provides a declarative approach:

1. **Mesh configuration** — The trainer config defines a logical device mesh with **named axes** and the number of devices along each axis. For example, a 32K-device mesh might be configured as `mesh_shape=(1, 16, 1, 256, 1, 8, 1)` with `mesh_axis_names=("pipeline", "data", "expert", "fsdp", "seq", "track", "model")`, using 16 devices for data parallelism, 256 for FSDP, and 8 for [track parallelism](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates) (16 × 256 × 8 = 32,768). Configs can also define `mesh_rules` to select different mesh shapes based on the hardware.

2. **Layer specs** — Each layer declares how its parameters should be sharded by referencing axis names (not sizes). For example, a weight matrix of shape `(4096, 1024)` with `mesh_axes=("fsdp", None)` will have its first dimension sharded across the 256 FSDP devices, so each device holds a `(16, 1024)` slice.

3. **Automatic collection** — Each layer's sharding declaration (including `mesh_axes`) is wrapped in a `ParameterSpec`. During initialization, the trainer traverses the model hierarchy (Model → Decoder → Attention → Linear, etc.) to gather all specs into a nested dict (pytree) that mirrors the model structure.

4. **pjit integration** — The collected specs are passed to JAX's `pjit`, which compiles SPMD code that automatically inserts collective operations (all-reduce, all-gather) and ensures each device only holds its shard of the parameters.

## 1. Configuring the Device Mesh

### 1.1 mesh_shape and mesh_axis_names

The `SpmdTrainer.Config` defines the device mesh via two fields ([trainer.py#L120-L122](https://github.com/apple/axlearn/blob/main/axlearn/common/trainer.py)):

```python
class SpmdTrainer(Module):
  class Config(Module.Config):
    mesh_shape: Required[Union[MeshShape, HybridMeshShape]] = REQUIRED
    mesh_axis_names: Required[Sequence[str]] = REQUIRED
```

For example, a config might specify:

```python
mesh_shape = (1, 16, 1, 256, 1, 8, 1)
mesh_axis_names = ("pipeline", "data", "expert", "fsdp", "seq", "track", "model")
```

This creates a 7D logical mesh where:
- `data` axis has 16 devices for data parallelism
- `fsdp` axis has 256 devices for fully-sharded data parallelism
- `track` axis has 8 devices for track parallelism (used in [PT-MoE architectures](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates))
- Other axes have size 1 (unused)

### 1.2 mesh_shape_from_axes Helper

Instead of writing raw tuples like `(1, 16, 1, 256, 1, 8, 1)` and remembering which position is which axis, AXLearn provides a helper with named arguments:

```python
mesh_shape_from_axes()                          # → (1, 1, 1, 1, 1, 1, 1)
mesh_shape_from_axes(data=16, fsdp=256, track=8)  # → (1, 16, 1, 256, 1, 8, 1)
mesh_shape_from_axes(data=-1, fsdp=256, track=8)  # → (1, -1, 1, 256, 1, 8, 1)
```

The special value `-1` means "infer at runtime". For instance, on 32K TPU chips with `mesh_shape_from_axes(data=-1, fsdp=256, track=8)`:

```
total  = pipeline × data × expert × fsdp × seq × track × model
32,768 = 1 × data × 1 × 256 × 1 × 8 × 1
data   = 32,768 / (256 × 8) = 16
```

Final mesh: `(1, 16, 1, 256, 1, 8, 1)`

### 1.3 mesh_rules: Hardware-Specific Overrides

Different hardware topologies may require different sharding strategies. The `mesh_rules` field allows pattern-based overrides ([trainer.py#L127-L137](https://github.com/apple/axlearn/blob/main/axlearn/common/trainer.py)):

```python
class SpmdTrainer(Module):
  class Config(Module.Config):
    mesh_rules: Optional[Sequence[tuple[str, Optional[MeshShape]]]] = None
```

Example configuration:

```python
mesh_shape = mesh_shape_from_axes(data=-1, fsdp=256, track=8),  # default for 32K
mesh_rules = (
    ("tpu-v5p-4096-8", mesh_shape_from_axes(data=-1, fsdp=256, track=8)),  # 8 slices × 4096 = 32K
    ("tpu-v5p-4096-2", mesh_shape_from_axes(data=-1, fsdp=128, track=4)),  # 2 slices × 4096 = 8K
    ("tpu-v6e-2048", mesh_shape_from_axes(data=-1, fsdp=64, track=1)),     # single slice = 4K
)
```

At launch time, a `--mesh_selector` flag (e.g., `--mesh_selector=tpu-v5p-4096-8`) is matched against these patterns using regex. The `select_mesh_config` function handles this ([trainer.py#L1357-L1379](https://github.com/apple/axlearn/blob/main/axlearn/common/trainer.py)):

```python
def select_mesh_config(trainer_config: SpmdTrainer.Config, *, mesh_selector: str):
    """Selects a mesh rule (if one matches) to override mesh config."""
    if trainer_config.mesh_rules:
        mesh_rule = match_regex_rules(
            mesh_selector, rules=trainer_config.mesh_rules, default_value=REQUIRED
        )
        if mesh_rule is not REQUIRED:
            trainer_config.mesh_shape = mesh_rule
```

### 1.4 Mesh Creation at Runtime

In `SpmdTrainer.__init__`, the mesh is created from the configuration:

```python
devices = utils.create_device_mesh(mesh_shape=cfg.mesh_shape)
mesh = jax.sharding.Mesh(devices, cfg.mesh_axis_names)
```

This produces a `jax.sharding.Mesh` object that maps physical devices to the named logical axes.

## 2. Declaring Parameter Sharding

### 2.1 ParameterSpec

AXLearn uses `ParameterSpec` to describe layer parameters. It inherits from `TensorSpec` ([utils.py](https://github.com/apple/axlearn/blob/main/axlearn/common/utils.py)) and adds training-specific fields ([base_layer.py](https://github.com/apple/axlearn/blob/main/axlearn/common/base_layer.py)):

```python
@dataclasses.dataclass
class TensorSpec:
    shape: Sequence[int]
    dtype: Optional[jnp.dtype] = None
    mesh_axes: Optional[jax.sharding.PartitionSpec] = None  # sharding spec

@dataclasses.dataclass
class ParameterSpec(TensorSpec):
    ...  # initializer, factorization, fan_axes, weight_decay_scale
```

> **Historical note:** AXLearn development began in February 2022, when JAX's sharding APIs were still experimental (`jax.experimental.pjit`). JAX 0.4.1 (December 2022) later introduced stable APIs like `jax.Array` and `NamedSharding`. AXLearn's `ParameterSpec` and `TensorSpec` were designed to insulate training code from these evolving APIs, while also bundling sharding with training-specific metadata (initializer, factorization, weight decay) that JAX's `PartitionSpec` doesn't handle.

The key field for sharding is `mesh_axes`, which references the named axes from `mesh_axis_names`. For example:

- `mesh_axes=("fsdp", "model")` — shard dim 0 across `fsdp`, dim 1 across `model`
- `mesh_axes=(None, "model")` — replicate dim 0, shard dim 1 across `model`
- `mesh_axes=(None, None)` — fully replicate the parameter

### 2.2 Layers Implement _create_layer_parameter_specs()

Each layer declares its parameters by overriding `_create_layer_parameter_specs()`. Here's the `Linear` layer ([layers.py#L739-L757](https://github.com/apple/axlearn/blob/main/axlearn/common/layers.py)):

```python
class Linear(DenseGeneralBaseLayer):
    @config_class
    class Config(DenseGeneralBaseLayer.Config):
        input_dim: Required[int] = REQUIRED
        output_dim: Required[int] = REQUIRED
        bias: bool = True

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None)  # default: replicate
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim],
                mesh_axes=(cfg.param_partition_spec[-1],),
            )
        return params
```

The sharding is configurable via `param_partition_spec`. Model builders can override this:

```python
linear_cfg = Linear.default_config().set(
    input_dim=4096,
    output_dim=1024,
    param_partition_spec=("fsdp", "model"),  # shard weight across fsdp and model axes
)
```

### 2.3 Recursive Collection: create_parameter_specs_recursively()

The trainer collects all parameter specs by calling `create_parameter_specs_recursively()` on the model ([base_layer.py#L610-L642](https://github.com/apple/axlearn/blob/main/axlearn/common/base_layer.py)):

```python
def create_parameter_specs_recursively(self) -> NestedParameterSpec:
    specs: dict[str, NestedParameterSpec] = {}
    
    # Get this layer's direct parameters
    param_specs = self._create_layer_parameter_specs()
    for name, param_spec in param_specs.items():
        partition_spec = param_spec.mesh_axes
        if partition_spec is None:
            partition_spec = [None] * len(param_spec.shape)
        param_spec = dataclasses.replace(
            param_spec,
            mesh_axes=PartitionSpec(*partition_spec),
        )
        specs[name] = param_spec
    
    # Recursively get children's parameters
    for name, child in self._children.items():
        if isinstance(child, BaseLayer):
            specs[name] = child.create_parameter_specs_recursively()
    
    return specs
```

This traverses the entire model hierarchy:

```
Model
├── decoder
│   └── transformer
│       └── layer[0..N]
│           ├── self_attention
│           │   └── attention
│           │       ├── q_proj → _create_layer_parameter_specs()
│           │       ├── k_proj → _create_layer_parameter_specs()
│           │       └── ...
│           └── feed_forward
│               ├── linear1 → _create_layer_parameter_specs()
│               └── linear2 → _create_layer_parameter_specs()
└── lm_head → _create_layer_parameter_specs()
```

The result is a nested dict matching the model structure:

```python
{
    "decoder": {
        "transformer": {
            "layer0": {
                "self_attention": {
                    "attention": {
                        "q_proj": {"weight": ParameterSpec(...), "bias": ...},
                        "k_proj": {"weight": ParameterSpec(...), ...},
                    }
                },
                "feed_forward": {
                    "linear1": {"weight": ParameterSpec(...), ...},
                }
            },
        }
    },
    "lm_head": {"weight": ParameterSpec(...), ...}
}
```

## 3. From ParameterSpec to JAX Sharding

### 3.1 ParameterSpec vs PartitionSpec

- **`ParameterSpec`** — AXLearn's rich abstraction containing shape, dtype, mesh_axes, initializer, factorization, etc.
- **`PartitionSpec`** — JAX's native sharding type (`jax.sharding.PartitionSpec`), just a tuple of axis names

JAX's SPMD system only understands `PartitionSpec`. AXLearn uses `ParameterSpec` internally for its additional metadata, then extracts the sharding info for JAX.

### 3.2 Extracting the Sharding Tree

In `SpmdTrainer.__init__`:

```python
self._model_param_specs = self.model.create_parameter_specs_recursively()

model_param_partition_specs = jax.tree.map(
    lambda spec: spec.mesh_axes, self._model_param_specs
)
```

This transforms the tree of `ParameterSpec` into a tree of `PartitionSpec`:

```python
# Before: nested dict of ParameterSpec
{
    "linear": {
        "weight": ParameterSpec(shape=(4096, 1024), 
                                mesh_axes=PartitionSpec("fsdp", "model"), ...),
        "bias": ParameterSpec(shape=(1024,), 
                              mesh_axes=PartitionSpec("model"), ...)
    }
}

# After: nested dict of PartitionSpec
{
    "linear": {
        "weight": PartitionSpec("fsdp", "model"),
        "bias": PartitionSpec("model")
    }
}
```

## 4. Applying Sharding via pjit

### 4.1 The _pjit_train_step Method

The partition specs are passed to `pjit` (partitioned JIT) which tells JAX how to distribute inputs and outputs ([trainer.py#L1174-L1190](https://github.com/apple/axlearn/blob/main/axlearn/common/trainer.py)):

```python
def _pjit_train_step(self) -> jax.stages.Wrapped:
    return pjit(
        self._train_step,
        in_shardings=(
            self._trainer_state_partition_specs,  # sharding for trainer state
            self._train_step_input_partition_specs(),  # sharding for input batch
        ),
        out_shardings=(
            self._trainer_state_partition_specs,  # output state has same sharding
            dict(summaries=None, loss=None, aux=None),
        ),
        donate_argnums=(0,),
    )
```

The `_trainer_state_partition_specs` is a pytree of `PartitionSpec` with the same structure as `TrainerState`:

```python
TrainerState(
    prng_key=PartitionSpec(None),  # replicated
    model={...nested PartitionSpecs...},
    learner={...nested PartitionSpecs...},
)
```

### 4.2 What JAX Does With This

When `pjit` compiles the function:

1. **Shards inputs** — Each device receives only its portion of the data
2. **Generates SPMD code** — The same code runs on every device
3. **Inserts collectives** — Automatically adds `all-reduce`, `all-gather`, etc. where needed
4. **Shards outputs** — Results are distributed according to `out_shardings`

### 4.3 Concrete Example

For a weight with `PartitionSpec("fsdp", None)` on a mesh with `fsdp=256`:

```
Full weight shape: [4096, 1024]
Each device holds: [4096/256, 1024] = [16, 1024]

Device 0:   weight[0:16, :]
Device 1:   weight[16:32, :]
...
Device 255: weight[4080:4096, :]
```

During the forward pass:
- Each device computes with its local shard
- JAX inserts `all-gather` to collect activations when needed

During the backward pass:
- Each device computes gradients for its shard  
- JAX inserts `all-reduce` to synchronize gradients across the `fsdp` axis
- Each device updates only its shard

## 5. End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Configuration Time                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. Define mesh_shape and mesh_axis_names in trainer config         │
│  2. Optionally define mesh_rules for hardware-specific overrides    │
│  3. Layers define param_partition_spec referencing axis names       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Launch Time                                 │
├─────────────────────────────────────────────────────────────────────┤
│  1. --mesh_selector flag specifies hardware (e.g., "tpu-v5p-4096-8")│
│  2. select_mesh_config() matches against mesh_rules                 │
│  3. mesh_shape is finalized (with -1 inferred from device count)    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Trainer Initialization                         │
├─────────────────────────────────────────────────────────────────────┤
│  1. create_device_mesh() arranges physical devices into mesh        │
│  2. jax.sharding.Mesh() assigns axis names                          │
│  3. model.create_parameter_specs_recursively() collects all specs   │
│  4. jax.tree.map extracts PartitionSpec tree                        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Loop                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. pjit(_train_step) compiles with in_shardings/out_shardings      │
│  2. JAX generates SPMD code with collective operations              │
│  3. Each device holds shards and computes on local data             │
│  4. Collectives synchronize gradients and activations as needed     │
└─────────────────────────────────────────────────────────────────────┘
```

## See Also

- [mesh.md](mesh.md) — JAX device mesh and `PartitionSpec` basics
- [dp+fsdp+tp/](dp+fsdp+tp/) — Diagrams comparing parallelism strategies
- [AXLearn GitHub](https://github.com/apple/axlearn) — Source code
- [Apple Intelligence Foundation Language Models Tech Report](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025) — PT-MoE architecture details
