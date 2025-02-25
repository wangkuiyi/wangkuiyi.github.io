# JAX’s Device Mesh and Tensor Sharding

For distributed computing, JAX organizes GPUs or TPUs into a mesh.  It then shards a tensor across devices by overlaying its elements onto this mesh.  The following sample program illustrates the concept of a device mesh and tensor sharding.

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

assert len(jax.devices()) == 8

dv = jax.experimental.mesh_utils.create_device_mesh((4, 2))
print(type(dv))
print(dv)
```

When you run the program on a host with 8 GPUs, the output is:

```
<class 'numpy.ndarray'>
[[CudaDevice(id=0) CudaDevice(id=1)]
 [CudaDevice(id=2) CudaDevice(id=3)]
 [CudaDevice(id=4) CudaDevice(id=5)]
 [CudaDevice(id=6) CudaDevice(id=7)]]
```

This output indicates that **a mesh is simply an array with elements of type `CudaDevice`**.

If we change the shape of the mesh from 4×2 to 2×4, as shown below:

```python
dv = jax.experimental.mesh_utils.create_device_mesh((2, 4))
print(type(dv))
print(dv)
```

The output becomes:

```
<class 'numpy.ndarray'>
[[CudaDevice(id=0) CudaDevice(id=1) CudaDevice(id=2) CudaDevice(id=3)]
 [CudaDevice(id=4) CudaDevice(id=5) CudaDevice(id=6) CudaDevice(id=7)]]
```

This demonstrates that **the mesh is created by placing the devices into the array in row-major order**.

Before using the array, we need to convert it into a `jax.sharding.Mesh`.  This conversion allows us to assign names to each axis of the array.  In the example below, axis 0 is named "model" and axis 1 is named "data".

```python
mesh = jax.sharding.Mesh(dv, axis_names=("model", "data"))
print(mesh)  # Mesh('model': 2, 'data': 4)
```

With the mesh defined, we can now shard a tensor by mapping it onto the mesh.  The following example overlays an 8×8 tensor onto the 2×4 mesh.

```python
x = jnp.ones((8, 8))
x_sharded = jax.device_put(
    x,
    device=jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("model", "data"),
    ),
)
jax.debug.visualize_array_sharding(x_sharded)
```

Calling `jax.debug.visualize_array_sharding` displays how the tensor’s elements are distributed across the devices:

```
  GPU 0    GPU 1    GPU 2    GPU 3

  GPU 4    GPU 5    GPU 6    GPU 7
```

The names specified in the `jax.sharding.PartitionSpec` determine how the tensor’s axes are mapped to the mesh’s named axes.  In this example, the first axis (axis 0) of tensor x is mapped to the mesh’s "model" axis, and the second (axis 1) is mapped to the "data" axis.  Consequently, axis 0 of x corresponds to the two rows of the mesh, and axis 1 corresponds to the four columns.

We can also try switching the mapping:

```python
x = jnp.ones((8, 8))
x_sharded = jax.device_put(
    x,
    device=jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("data", "model"),  # Swaps the mapping: model -> data, data -> model
    ),
)
jax.debug.visualize_array_sharding(x_sharded)
```

Now the sharding output is as follows, because axis 0 of x is mapped to the "data" axis of the mesh (width 4) and axis 1 is mapped to the "model" axis (width 2):

```
   GPU 0       GPU 4

   GPU 1       GPU 5

   GPU 2       GPU 6

   GPU 3       GPU 7
```
