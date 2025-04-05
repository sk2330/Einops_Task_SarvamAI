import numpy as np
import re
from typing import List, Dict, Tuple, Union

def rearrange(tensor: Union[np.ndarray, List[np.ndarray]], pattern: str, **axes_lengths) -> np.ndarray:
    # Converting list to np array
    if isinstance(tensor, list):
        tensor = np.stack(tensor)

    # Parsing the --> pattern
    if '->' not in pattern:
        raise ValueError(f"Invalid pattern: {pattern}. Pattern must contain '->'")

    input_pattern, output_pattern = pattern.split('->')
    input_pattern = input_pattern.strip()
    output_pattern = output_pattern.strip()

    # Parsing the input and output patterns
    input_axes, input_composition = parse_axes(input_pattern)
    output_axes, output_composition = parse_axes(output_pattern)

    # Collect axis dimensions
    axes_dims = get_axes_dimensions(tensor, input_axes, input_composition, axes_lengths)

   ### Tensor operations
    return apply_operations(tensor, input_axes, output_axes,
                           input_composition, output_composition, axes_dims)


def parse_axes(pattern: str) -> Tuple[List[str], Dict[str, List[str]]]:
    ### Handling ellipsis
    pattern = pattern.replace('...', ' ... ')

    # Extract parenthesized parts (composite axes)
    paren_pattern = re.compile(r'\([^()]+\)')
    axes = []
    composition = {}

    # Replace parenthesized parts with placeholders
    pattern_with_placeholders = pattern
    placeholders_map = {}

    for match in paren_pattern.finditer(pattern):
        paren_expr = match.group(0)
        placeholder = f'__PAREN_{len(placeholders_map)}__'
        pattern_with_placeholders = pattern_with_placeholders.replace(paren_expr, placeholder)
        placeholders_map[placeholder] = paren_expr

        # Extract components
        components = paren_expr[1:-1].split()
        if not components:
            raise ValueError(f"Empty parenthesized expression in pattern: {pattern}")
        composition[paren_expr] = components

    for token in pattern_with_placeholders.split():
        if token.startswith('__PAREN_'):
            original_expr = placeholders_map[token]
            axes.append(original_expr)
        else:
            axes.append(token)

    # Remove empty tokens
    axes = [axis for axis in axes if axis]

    return axes, composition

def get_axes_dimensions(
    tensor: np.ndarray,
    input_axes: List[str],
    input_composition: Dict[str, List[str]],
    axes_lengths: Dict[str, int]
) -> Dict[str, int]:

    tensor_shape = tensor.shape
    axes_dims = {}

    # Handling ellipsis if present
    ellipsis_idx = -1
    if '...' in input_axes:
        ellipsis_idx = input_axes.index('...')

        # Calculate dimensions for ellipsis
        non_ellipsis_dims = len(input_axes) - 1  # -1 for ellipsis itself
        if non_ellipsis_dims > len(tensor_shape):
            raise ValueError(f"Tensor has {len(tensor_shape)} dimensions, but pattern requires at least {non_ellipsis_dims}")

        ellipsis_dims = len(tensor_shape) - non_ellipsis_dims

        # Add ellipsis dimensions
        for i in range(ellipsis_dims):
            axis_name = f'__ellipsis_{i}__'
            axes_dims[axis_name] = tensor_shape[ellipsis_idx + i]

        # Assign dimensions to non-ellipsis axes
        for i, axis in enumerate(input_axes):
            if axis == '...':
                continue

            tensor_idx = i if i < ellipsis_idx else i + ellipsis_dims - 1
            if tensor_idx >= len(tensor_shape):
                raise ValueError(f"Tensor shape {tensor_shape} doesn't match pattern {input_axes}")

            if axis in input_composition:
                # Composite axis
                axes_dims[axis] = tensor_shape[tensor_idx]
            else:
                # Elementary axis
                axes_dims[axis] = tensor_shape[tensor_idx]
    else:
        if len(input_axes) != len(tensor_shape):
            raise ValueError(f"Tensor has {len(tensor_shape)} dimensions, but pattern specifies {len(input_axes)}")

        #### Assigning dimensions to axes
        for i, axis in enumerate(input_axes):
            if axis in input_composition:
                #### Composite axis
                axes_dims[axis] = tensor_shape[i]
            else:
                #### Elementary axis
                axes_dims[axis] = tensor_shape[i]

    # Process composite axes (splitting)
    for composite_axis, components in input_composition.items():
        composite_dim = axes_dims[composite_axis]

        # Get dimensions for provided components
        known_components = {comp: axes_lengths[comp] for comp in components if comp in axes_lengths}

        # Calculate unknown component dimensions
        unknown_components = [comp for comp in components if comp not in known_components]

        if len(unknown_components) > 1:
            raise ValueError(f"Multiple unspecified dimensions in {composite_axis}: {unknown_components}")
        elif len(unknown_components) == 1:
            # Calculate the unknown dimension
            known_product = 1
            for dim in known_components.values():
                known_product *= dim

            if composite_dim % known_product != 0:
                raise ValueError(f"Cannot split {composite_axis} of size {composite_dim} evenly by {known_product}")

            #### Set the unknown dimension
            axes_dims[unknown_components[0]] = composite_dim // known_product
        elif len(unknown_components) == 0:
            #### Verify that product equals the composite dimension
            known_product = 1
            for dim in known_components.values():
                known_product *= dim

            if known_product != composite_dim:
                raise ValueError(f"Product of components {known_product} does not match {composite_axis} dimension {composite_dim}")

        # Add component dimensions
        for comp, dim in known_components.items():
            axes_dims[comp] = dim

    # Add any remaining axes_lengths
    for axis, length in axes_lengths.items():
        if axis not in axes_dims:
            # Check if it's a component of any composite axes
            is_component = False
            for components in input_composition.values():
                if axis in components:
                    is_component = True
                    break

            if not is_component:
                # Extra axis, might be used in output
                axes_dims[axis] = length

    return axes_dims

def apply_operations(
    tensor: np.ndarray,
    input_axes: List[str],
    output_axes: List[str],
    input_composition: Dict[str, List[str]],
    output_composition: Dict[str, List[str]],
    axes_dims: Dict[str, int]
) -> np.ndarray:

    # 1. Expand composite axes in input
    expanded_input_axes = []
    for axis in input_axes:
        if axis == '...':
            #### Handling ellipsis
            ellipsis_count = sum(1 for key in axes_dims if key.startswith('__ellipsis_'))
            expanded_input_axes.extend([f'__ellipsis_{i}__' for i in range(ellipsis_count)])
        elif axis in input_composition:
            # Expand composite axis
            expanded_input_axes.extend(input_composition[axis])
        else:
            # Regular axis
            expanded_input_axes.append(axis)

    ##### 2. Calculate reshape dimensions for input tensor
    reshape_dims = []
    for axis in expanded_input_axes:
        if axis not in axes_dims:
            raise ValueError(f"Missing dimension for axis {axis}")
        reshape_dims.append(axes_dims[axis])

    ##### Reshape to split composite axes
    if tuple(reshape_dims) != tensor.shape:
        try:
            tensor = tensor.reshape(reshape_dims)
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor from {tensor.shape} to {reshape_dims}: {e}")

    ##### 3. Expand composite axes in output
    expanded_output_axes = []
    for axis in output_axes:
        if axis == '...':
            ### Handle ellipsis
            ellipsis_count = sum(1 for key in axes_dims if key.startswith('__ellipsis_'))
            expanded_output_axes.extend([f'__ellipsis_{i}__' for i in range(ellipsis_count)])
        elif axis in output_composition:
            #### Expand composite axis
            expanded_output_axes.extend(output_composition[axis])
        else:
            #### Regular axis
            expanded_output_axes.append(axis)

    #### 4. Create mapping from expanded input axes to positions
    axes_positions = {axis: i for i, axis in enumerate(expanded_input_axes)}

    #### 5. Calculate transposition order
    transpose_order = []
    for axis in expanded_output_axes:
        if axis in axes_positions:
            transpose_order.append(axes_positions[axis])
        else:
            #### For repeating axes (not fully implemented)
            if axis in axes_dims:
                raise NotImplementedError(f"Repeating of axes is not fully implemented")
            else:
                raise ValueError(f"Unknown axis {axis} in output pattern")

    #### Apply transposition if needed
    if len(transpose_order) > 0 and tuple(transpose_order) != tuple(range(len(expanded_input_axes))):
        try:
            tensor = np.transpose(tensor, transpose_order)
        except ValueError as e:
            raise ValueError(f"Cannot transpose tensor with shape {tensor.shape} using order {transpose_order}: {e}")

    #### 6. Calculate output shape for merging
    output_shape = []
    for axis in output_axes:
        if axis == '...':
            #### Handle ellipsis
            ellipsis_count = sum(1 for key in axes_dims if key.startswith('__ellipsis_'))
            for i in range(ellipsis_count):
                output_shape.append(axes_dims[f'__ellipsis_{i}__'])
        elif axis in output_composition:
            #### Calculate composite dimension
            composite_dim = 1
            for component in output_composition[axis]:
                if component not in axes_dims:
                    raise ValueError(f"Missing dimension for axis {component} in output pattern")
                composite_dim *= axes_dims[component]
            output_shape.append(composite_dim)
        else:
            #### Regular axis
            if axis not in axes_dims:
                raise ValueError(f"Unknown axis in output pattern: {axis}")
            output_shape.append(axes_dims[axis])

    #### Reshape to merge dimensions in output
    if tuple(output_shape) != tensor.shape:
        try:
            tensor = tensor.reshape(output_shape)
        except ValueError as e:
            raise ValueError(f"Cannot reshape tensor from {tensor.shape} to {output_shape}: {e}")

    return tensor


