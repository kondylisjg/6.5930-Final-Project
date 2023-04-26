import torch
import torch.nn as nn
from loaders import *

def get_energy(model, x, hw_arch="todo.yaml"):
    layer_types, layer_params, data_sizes = _get_layers_and_input_shapes(model, x)

    # create dir with specs

    # sum up per layer energy
    total_energy = 0
    per_layer_energy = []

    for type, params, shape in zip(layer_types, layer_params, data_sizes):
        # get layer and map spec
        if type == "linear":
            layer_spec = _get_linear_layer_yaml(shape[0], params[0], params[1])
            map_spec = _get_linear_layer_yaml(shape[0], params[0], params[1])

        # write specs to file

        # execute timeloop
        stats, _ = run_timeloop_model(
            ConfigRegistry.SINGLE_PE_ARCH, ConfigRegistry.SINGLE_PE_COMPONENTS_DIR,
            ConfigRegistry.SINGLE_PE_MAP,
            ConfigRegistry.SMALL_LAYER_PROB
        )

        # parse out total energy in uJ
        layer_energy = stats.split("\n")[-20].split(" ")[1]
        total_energy += layer_energy
        per_layer_energy.append(layer_energy)

    return total_energy, per_layer_energy


def _get_linear_layer_yaml(batch_size, in_dim, out_dim):
    layers_spec = """
problem:
  shape:
    name: "linear"
    dimensions: [ B, I, O ]
    data-spaces:
    - name: Weights
      projection:
      - [ [I] ]
      - [ [O] ]
    - name: Inputs
      projection:
      - [ [B] ]
      - [ [I] ]
    - name: Outputs
      projection:
      - [ [B] ]
      - [ [O] ]
      read-write: True

  instance:
    B: {0}  # batch size
    I: {1}  # in dim
    O: {2}  # out dim
""".format(batch_size, in_dim, out_dim)

    return layers_spec


def _get_linear_map_yaml():
    return

def _get_layers_and_input_shapes(model, x):

    data_sizes = [x.shape]
    layer_types = []
    layer_params = []

    for module in model.modules():
        if isinstance(module, nn.Linear):
            layer_types.append("linear")
            layer_params.append((module.in_features, module.out_features))
        elif isinstance(module, nn.Conv2d):
            print(module)
        elif isinstance(module, nn.Conv3d):
            print(module)
        elif isinstance(module, nn.Sequential):
            # TODO: Same for RELU and all activations
            continue
        else:
            raise NotImplementedError(f"{module} layer is not supported by get_energy")

        x = module(x)
        data_sizes.append(x.shape)

    return layer_types, layer_params, data_sizes


if __name__ == "__main__":

    model = nn.Sequential(
        nn.Linear(764, 100),
        nn.Linear(100, 50),
        nn.Linear(50, 10),
    )

    x = torch.rand(1, 764)

    get_energy(model, x)


