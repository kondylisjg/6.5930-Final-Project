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
    for n, (type, params, x_in, x_out) in enumerate(zip(layer_types, layer_params, data_sizes, data_sizes[1:])):
        # get yaml strings for layer shape and mapping
        if type == "linear":
            layer_spec = _get_linear_layer_yaml(x_in[0], params[0], params[1])
            map_spec = _get_linear_map_yaml(x_in[0], params[0], params[1])
        elif type == "conv2d":
            assert x_in[1] == params[0]
            assert x_out[1] == params[1]
            layer_spec = _get_conv2d_layer_yaml(x_in[0], params[0], params[1], params[2][0], params[2][1], x_out[1], x_out[2], params[3][0], params[3][1])
            map_spec = _get_conv2d_map_yaml(x_in[0], x_in[-1], x_out[-1], params[0], params[1], x_out[1], x_out[2])
        elif type == "conv3d":
            assert x_in[1] == params[0]
            assert x_out[1] == params[1]
            layer_spec = _get_conv3d_layer_yaml(x_in[0], params[0], params[1], params[2][0], params[2][1], params[2][2], x_out[2], x_out[3], x_out[4], params[3][0], params[3][1], params[3][2])
            map_spec = _get_conv3d_map_yaml(x_in[0], params[0], params[1], params[2][0], params[2][1], params[2][2], x_out[2], x_out[3], x_out[4])
        else:
            raise NotImplementedError

        # write specs to file
        layer_shape_path = os.path.join(layers_path, f"layer_shape_{n}.yaml")
        map_path = os.path.join(layers_path, f"map_{n}.yaml")
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

        # debug
        if verbose:
            print()
            print("############################################################")
            print(n, "Layer:", type, params, "input dim:", shape)
            print("############################################################")
            print(stats)

        # parse out total energy in uJ
        layer_energy = float(stats.split("\n")[-20].split(" ")[1])
        total_energy += layer_energy
        per_layer_energy.append(layer_energy)

    return total_energy, per_layer_energy


def _get_linear_layer_yaml(batch_size, in_dim, out_dim):

    layers_spec = f"""
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
    B: {batch_size}  # batch size
    I: {in_dim}  # in dim
    O: {out_dim}  # out dim"""

    return layers_spec


def _get_linear_map_yaml(batch_size, in_dim, out_dim):
    raise NotImplementedError


def _get_conv2d_layer_yaml(batch_size,
                           in_channel,
                           out_channel,
                           filter_height,
                           filter_width,
                           output_height,
                           output_width,
                           w_stride,
                           h_stride):

    layer_spec = f"""
problem:
  shape:
    name: "Conv2D"
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
    C: {in_channel}  # inchn
    M: {out_channel}  # outchn
    R: {filter_height}   # filter height
    S: {filter_width}   # filter width
    P: {output_height}   # ofmap height
    Q: {output_width}   # ofmap width
    N: {batch_size}   # batch size
    Wstride: {w_stride}
    Hstride: {h_stride}"""

    return layer_spec

def _get_conv2d_map_yaml(batch_size,
                         in_channel,
                         out_channel,
                         filter_height,
                         filter_width,
                         output_height,
                         output_width):

    map_spec = f"""
mapping:
  # mapping for the DRAM
  - target: DRAM
    type: temporal
    factors: R={filter_height} S={filter_width} P={output_height} Q={output_width} N={batch_size} M={out_channel} C={in_channel}
    permutation:
  # mapping for the local scratchpad inside the PE
  - target: scratchpad
    type: temporal
    factors:
    permutation:
  - target: scratchpad
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  # mapping for the input and output registers of the mac unit
  - target: weight_reg
    type: temporal
    factors:
    permutation:
  - target: weight_reg
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  - target: input_activation_reg
    type: temporal
    factors:
    permutation:
  - target: input_activation_reg
    type: bypass
    keep: [Inputs]
    bypass: [Weights, Outputs]
  - target: output_activation_reg
    type: temporal
    factors:
    permutation:
  - target: output_activation_reg
    type: bypass
    keep: [Outputs]
    bypass: [Weights, Inputs]"""

    return map_spec


def _get_conv3d_layer_yaml(batch_size,
                           in_channel,
                           out_channel,
                           filter_height,
                           filter_width,
                           filter_depth,
                           output_height,
                           output_width,
                           output_depth,
                           w_stride,
                           h_stride,
                           d_stride):

    layer_spec = f"""
problem:
  shape:
    name: "Conv3D"
    dimensions: [ R, S, T, P, Q, F, C, M, N ]
    coefficients:
    - name: Wstride
      default: 1
    - name: Hstride
      default: 1
    - name: Dstride
      default: 1
    - name: Wdilation
      default: 1
    - name: Hdilation
      default: 1
    - name: Ddilation
      default: 1

    data-spaces:
    - name: Weights
      projection:
      - [ [C] ]
      - [ [M] ]
      - [ [R] ]
      - [ [S] ]
      - [ [T] ]
    - name: Inputs
      projection:
      - [ [N] ]
      - [ [C] ]
      - [ [R, Wdilation], [P, Wstride] ] # SOP form: RWdilation + PWstride
      - [ [S, Hdilation], [Q, Hstride] ] # SOP form: SHdilation + QHstride
      - [ [T, Ddilation], [F, Dstride] ] # SOP form: TDdilation + FDstride
    - name: Outputs
      projection:
      - [ [N] ]
      - [ [M] ]
      - [ [Q] ]
      - [ [P] ]
      - [ [F] ]
      read-write: True
   
  instance:  
    N: {batch_size} # batch size
    C: {in_channel} # in channels (number of kernels)
    M: {out_channel} # out channels (number of kernels)
    R: {filter_width} # filter width
    S: {filter_height} # filter height
    T: {filter_depth} # filter depth
    Q: {output_width} # output image width (x)
    P: {output_height} # output image height (y)
    F: {output_depth} # output image depth (z)
    Wstride: {w_stride}
    Hstride: {h_stride}
    Dstride: {d_stride}"""

    return layer_spec


def _get_conv3d_map_yaml(batch_size,
                         in_channel,
                         out_channel,
                         filter_height,
                         filter_width,
                         filter_depth,
                         output_height,
                         output_width,
                         output_depth):

    map_spec = f"""
mapping:
  # mapping for the DRAM
  - target: DRAM
    type: temporal
    factors: N={batch_size} C={in_channel} M={out_channel} R={filter_width} S={filter_height} T={filter_depth} Q={output_width} P={output_height} F={output_depth}   
    permutation:
  # mapping for the local scratchpad inside the PE
  - target: scratchpad
    type: temporal
    factors: # factor of 0 => full dimension
    permutation:
  - target: scratchpad
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  # mapping for the input and output registers of the mac unit
  - target: weight_reg
    type: temporal
    factors:
    permutation: 
  - target: weight_reg
    type: bypass
    keep: [Weights]
    bypass: [Inputs, Outputs]
  - target: input_activation_reg
    type: temporal
    factors:
    permutation: 
  - target: input_activation_reg
    type: bypass
    keep: [Inputs]
    bypass: [Weights, Outputs]
  - target: output_activation_reg
    type: temporal
    factors:
    permutation: 
  - target: output_activation_reg
    type: bypass
    keep: [Outputs]
    bypass: [Weights, Inputs]"""

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
            layer_types.append("conv2d")
            layer_params.append((module.in_channels, module.out_channels, module.kernel_size, module.stride))
        elif isinstance(module, nn.Conv3d):
            print(module)
            layer_types.append("conv3d")
            layer_params.append((module.in_channels, module.out_channels, module.kernel_size, module.stride))
        elif isinstance(module, nn.Sequential):
            continue
        elif not (False):
            # activation functions and pooling layers: just apply but we don't measure their energy
            raise NotImplementedError(f"{module} layer is not supported by get_energy")

        x = module(x)
        data_sizes.append(x.shape)

    print(data_sizes)

    return layer_types, layer_params, data_sizes


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv3d(
                1, 5, (2, 2, 1), stride=(1, 1, 1))
    )

    x = torch.rand((1, 1, 100, 100, 4))

    _get_layers_and_input_shapes(model, x)

    # total, layerwise = get_energy(model, x)
    # print(total)
    # print(layerwise)


