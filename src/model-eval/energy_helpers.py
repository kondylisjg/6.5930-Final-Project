from pathlib import PosixPath

import torch
import torch.nn as nn
import os
from src.common.loaders import run_timeloop_model


def get_energy(model,
               x,
               hw_arch_path="singlePE/arch/single_PE_arch.yaml",
               hw_components_dir_path="singlePE/arch/components",
               layers_path="layers",
               verbose=False):

    # TODO: if accelergy estimates not done yet, do now
    # subprocess.run("accelergyTables - r / home / workspace / lab3 / PIM_estimation_tables")

    x = x.detach().clone()

    # get model and input shapes
    layer_types, layer_params, data_sizes = _get_layers_and_input_shapes(model, x)

    # create dir with layer shapes and mapping constraints
    if not os.path.exists(layers_path):
        os.makedirs(layers_path)

    # sum up per layer energy
    total_energy = 0
    per_layer_energy = []
    for n, (type, params, shape) in enumerate(zip(layer_types, layer_params, data_sizes)):
        # get layer and map spec
        if type == "linear":
            layer_spec = _get_linear_layer_yaml(shape[0], params[0], params[1])
            map_spec = _get_linear_map_yaml(shape[0], params[0], params[1])
        else:
            raise NotImplementedError

        # write specs to file
        layer_shape_path = os.path.join(layers_path, f"layer_shape_{n}.yaml")
        map_path = os.path.join(layers_path, f"map_{n}.yaml")
        # layer_shape_path = os.path.join(layers_path, "tiny_layer.yaml")
        # map_path = os.path.join(layers_path, "map.yaml")
        with open(layer_shape_path, "w+") as f:
            f.write(layer_spec)
        with open(map_path, "w+") as f:
            f.write(map_spec)

        # execute timeloop
        hw_arch_path = os.path.join("../common", hw_arch_path)
        hw_components_dir_path = os.path.join("../common", hw_components_dir_path)
        hw_arch_path = PosixPath(hw_arch_path)
        hw_components_dir_path = PosixPath(hw_components_dir_path)
        layer_shape_path = PosixPath(layer_shape_path)
        map_path = PosixPath(map_path)
        stats, _ = run_timeloop_model(hw_arch_path, hw_components_dir_path, map_path, layer_shape_path)

        # parse out total energy in uJ
        if verbose:
            print()
            print("############################################################")
            print(n, "Layer:", type, params, "input dim:", shape)
            print("############################################################")
            print(stats)

        layer_energy = float(stats.split("\n")[-20].split(" ")[1])
        total_energy += layer_energy
        per_layer_energy.append(layer_energy)

    return total_energy, per_layer_energy


def _get_linear_layer_yaml(batch_size, in_dim, out_dim):
    layer_spec = """
problem:
  shape:
    name: "CNN-Layer"
    dimensions: [ C, M, R, S, N, P, Q ]
    coefficients:
    - name: Wstride
      default: 1
    - name: Hstride
      default: 1
    - name: Wdilation
      default: 1
    - name: Hdilation
      default: 1

    data-spaces:
    - name: Weights
      projection:
      - [ [C] ]
      - [ [M] ]
      - [ [R] ]
      - [ [S] ]
    - name: Inputs
      projection:
      - [ [N] ]
      - [ [C] ]
      - [ [R, Wdilation], [P, Wstride] ] # SOP form: R*Wdilation + P*Wstride
      - [ [S, Hdilation], [Q, Hstride] ] # SOP form: S*Hdilation + Q*Hstride
    - name: Outputs
      projection:
      - [ [N] ]
      - [ [M] ]
      - [ [Q] ]
      - [ [P] ]
      read-write: True

  instance:
    C: 32  # inchn
    M: 64  # outchn
    R: 3   # filter height
    S: 3   # filter width
    P: 5   # ofmap height
    Q: 5   # ofmap width
    N: 2   # batch size"""

    return layer_spec




#     layers_spec = """
# problem:
#   shape:
#     name: "linear"
#     dimensions: [ B, I, O ]
#     data-spaces:
#     - name: Weights
#       projection:
#       - [ [I] ]
#       - [ [O] ]
#     - name: Inputs
#       projection:
#       - [ [B] ]
#       - [ [I] ]
#     - name: Outputs
#       projection:
#       - [ [B] ]
#       - [ [O] ]
#       read-write: True
#
#   instance:
#     B: {0}  # batch size
#     I: {1}  # in dim
#     O: {2}  # out dim
# """.format(batch_size, in_dim, out_dim)
#
#     return layers_spec


def _get_linear_map_yaml(batch_size, in_dim, out_dim):
    map_spec = """
mapping:
  # mapping for the DRAM
  - target: DRAM
    type: temporal
    factors: R=1 S=1 P=1 Q=1 N=2 M=32 C=32
    permutation: RSPQCMN
  # mapping for the local scratchpad inside the PE
  - target: scratchpad
    type: temporal
    factors: R=0 S=0 P=0 Q=0 N=1 M=2 C=1 # factor of 0 => full dimension
    permutation: QPNCMSR
  - target: scratchpad
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  # mapping for the input and output registers of the mac unit
  - target: weight_reg
    type: temporal
    factors: R=1 S=1 P=1 Q=1 M=1 C=1 N=1
    permutation: PQCMRSN
  - target: weight_reg
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  - target: input_activation_reg
    type: temporal
    factors: R=1 S=1 P=1 Q=1 M=1 C=1 N=1
    permutation: PQCMRSN
  - target: input_activation_reg
    type: bypass
    keep: [Inputs]
    bypass: [Weights, Outputs]
  - target: output_activation_reg
    type: temporal
    factors: R=1 S=1 P=1 Q=1 M=1 C=1 N=1
    permutation: PQCMRSN
  - target: output_activation_reg
    type: bypass
    keep: [Outputs]
    bypass: [Weights, Inputs]
"""
    return map_spec

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

    total, layerwise = get_energy(model, x)
    print(total)
    print(layerwise)


