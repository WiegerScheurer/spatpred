import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

def isotropic_gaussian(dims:tuple, sigma:float):
    x = np.arange(0, dims[0], 1, float)
    y = np.arange(0, dims[1], 1, float)
    x, y = np.meshgrid(x, y)
    x0 = dims[0] // 2
    y0 = dims[1] // 2
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def circle_stim(dims:tuple, radius:float, center:tuple=None):
    dot_img = np.zeros((dims[0], dims[1]))
    if center is None:
        center = (dims[0] // 2, dims[1] // 2)
    
    y, x = np.ogrid[-center[0]:dims[0]-center[0], -center[1]:dims[1]-center[1]]
    mask = x*x + y*y <= radius*radius

    dot_img[mask] = 1
    return dot_img

def random_dot(dims:tuple, n_dots:int, dot_rad:int):
    dot_img = np.zeros(dims)
    for i in range(n_dots):
        x = random.randint(dot_rad, dims[0]-dot_rad)
        y = random.randint(dot_rad, dims[1]-dot_rad)
        cv2.circle(dot_img, (y, x), dot_rad, (1,), -1)
    return dot_img

def show(input:np.ndarray, cmap:str='binary', figsize:tuple=(6,6), invert_y:bool=True):
    """
    Display an image using matplotlib.

    Parameters:
    input (np.ndarray): The input image.
    cmap (str): The color map to use for displaying the image. Default is 'binary'.
    figsize (tuple): The size of the figure. Default is (6,6).
    """
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(input, cmap=cmap)
    ax.axis("off")
    
    if invert_y:
        ax.invert_yaxis()  # Invert the y-axis
    plt.show()

def show_random_filter(pyramid, filter):
    # The indices for the filters that are within the patch
    filter_indices = np.where(filter == True)[0]
    this_filter = random.choice(filter_indices)
    pyramid.view.show_filter(this_filter)
    plt.show()

def cut_corners(input:np.ndarray, cut_size:int):
    """
    Applies a corner-cutting filter to the input array.

    Args:
        input (np.ndarray): The input array.
        cut_size (int): The size of the corner to be cut.

    Returns:
        np.ndarray: The filtered array with corners cut.
    """
    bool_mask = np.ones_like(input, dtype=bool)
    bool_mask[cut_size:-cut_size, cut_size:-cut_size] = False

    return input * ~bool_mask

def make_checker(dims, checkercenter, scales, scaling_factor, checker_size, stride):
    """
    Create a checkerboard pattern with optional scaling and centering.

    Args:
        dims (tuple): The dimensions of the checkerboard.
        checkercenter (tuple): The center coordinates of the checkerboard.
        scales (int): The number of scales for the checkerboard.
        scaling_factor (float): The scaling factor for each scale.
        checker_size (int): The size of each checkerboard square.
        stride (int): The stride for the smaller checkerboard.

    Returns:
        numpy.ndarray: The generated checkerboard pattern.
    """
    
    if scales == 0:
        # Create a full checkerboard of the given dimensions and checker size
        checkerboard = np.indices(dims).sum(axis=0) % 2
        checkerboard = np.repeat(checkerboard, checker_size, axis=0)
        checkerboard = np.repeat(checkerboard, checker_size, axis=1)
        checkerboard = checkerboard[:dims[0], :dims[1]].astype(float)
        return checkerboard
    else:
        # Create a checkerboard of the current scale
        checkerboard = np.indices(dims).sum(axis=0) % 2
        checkerboard = np.repeat(checkerboard, checker_size, axis=0)
        checkerboard = np.repeat(checkerboard, checker_size, axis=1)
        checkerboard = checkerboard[:dims[0], :dims[1]].astype(float)

        # Create a smaller checkerboard in the center
        smaller_dims = (int((dims[0] - 2 * stride) / scaling_factor), int((dims[1] - 2 * stride) / scaling_factor))
        smaller_checker_size = int(checker_size / scaling_factor)
        smaller_checkerboard = make_checker(smaller_dims, checkercenter, scales - 1, scaling_factor, smaller_checker_size, stride)

        # Replace the section of the larger checkerboard with the smaller one
        start = (dims[0] // 2 - smaller_dims[0] // 2, dims[1] // 2 - smaller_dims[1] // 2)
        checkerboard[start[0] + stride:start[0] + smaller_dims[0] + stride, start[1] + stride:start[1] + smaller_dims[1] + stride] = smaller_checkerboard

        return checkerboard

def plot_filter_locations(gabor_pyramid, in_range, pixdims=(425, 425), pix_per_dim:float=(425/8.4)):
    """
    Plots the locations of filters in a Gabor pyramid.

    Parameters:
    - gabor_pyramid: The Gabor pyramid object.
    - in_range: A boolean array indicating whether each filter is within the desired range.
    - pixdims: The dimensions of the image in pixels. Default is (425, 425).
    - pix_per_dim: The number of pixels per dimension. Default is 425/8.4.

    Returns:
    None
    """
    
    # Initialize an empty array with shape (nfilters, 2)
    coordinates = np.empty((gabor_pyramid.view.nfilters, 2))

    # Fill in the values
    for filter in range(gabor_pyramid.view.nfilters):
        coordinates[filter, 0] = gabor_pyramid.view.filters[filter]["centerh"] * (
            pixdims[0] / pix_per_dim
        )
        coordinates[filter, 1] = gabor_pyramid.view.filters[filter]["centerv"] * (
            pixdims[1] / pix_per_dim
        )

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the locations of all filters, making the ones outside the range transparent
    for i in range(gabor_pyramid.view.nfilters):
        if in_range[i]:
            ax.plot(coordinates[i, 0], coordinates[i, 1], 'o', color='red', alpha=0.1)
        else:
            ax.plot(coordinates[i, 0], coordinates[i, 1], 'o', color='blue', alpha=0.1)

    # Set the limits of the plot to the size of the image
    ax.set_xlim(0, pixdims[0] / pix_per_dim)
    ax.set_ylim(0, pixdims[1] / pix_per_dim)

    # Show the plot
    plt.show()
    
def plot_filter_outputs(sel_output, filters_per_freq_sel, spat_freqs, img_indices=[0, 1]):
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    start_idx = 0

    for i, ax in enumerate(axes.flat):
        for img in img_indices:
            ax.plot(sel_output[img][start_idx:start_idx+filters_per_freq_sel[i]])
        
        start_idx += filters_per_freq_sel[i]
        ax.set_title(f"Gabor filter pyramid output \nSpatial frequency {spat_freqs[i]/8.4} cycles per degree")

    plt.show()

def normalize_output(output, n_spatfreqs, filters_per_freq):
    """
    Normalizes the output of the Gabor pyramid.

    Parameters:
    - output: The output of the Gabor pyramid.

    Returns:
    - The normalized output.
    """
    start = 0
    circle_output_norm = []
    for i in range(n_spatfreqs):
        end = start + filters_per_freq[i]
        circle_output_norm.append(zs(output[start:end]))
        start = end

    
    return np.concatenate(circle_output_norm)
    
def select_filters(
    pyramid,
    output,
    imgs: np.ndarray,
    img_no: int,
    spat_freqs: list[float],
    filters_per_freq: list[float],
    percentile_cutoff: float = 99,
    plot: bool = False,
    verbose: bool = False,
):
    """
    Selects filters from a pyramid based on the given parameters.

    Args:
        pyramid: The pyramid object containing the filters.
        output: The output of the pyramid for the given image.
        imgs: The array of input images.
        img_no: The index of the image to select filters for.
        spat_freqs: The list of spatial frequencies.
        filters_per_freq: The list of number of filters per spatial frequency.
        percentile_cutoff: The percentile cutoff for filter selection (default: 99).
        plot: Whether to plot the selected filters (default: False).
        verbose: Whether to print verbose information (default: False).

    Returns:
        output_norm: The normalized output for each spatial frequency.
        filters_per_freq_sel: The number of selected filters per spatial frequency.
        filter_selection: The boolean mask for filter selection.
        filter_selection_dictlist: The list of selected filters as dictionary objects.
    """
    
    this_output=output[img_no]
    
    # output = gauss_output[img_no]
    if plot:
        show(imgs[img_no], figsize=(6, 6))

    n_spatfreqs = len(spat_freqs)

    # normalise the output for every spatial frequency separately
    start = 0
    output_norm = []
    for i in range(n_spatfreqs):
        end = start + filters_per_freq[i]
        output_norm.append(zs(this_output[start:end]))
        start = end

    # Calculate the nth percentile for each spatial frequency and create a boolean mask
    filter_selection = []
    filters_per_freq_sel = []
    for i in range(n_spatfreqs):
        percentile = np.percentile(output_norm[i], percentile_cutoff)
        mask = output_norm[i] > percentile
        filter_selection.append(mask)
        n_filters = np.sum(mask)
        filters_per_freq_sel.append(n_filters)
        if verbose:
            print(
                f"Spatial frequency {i}: percentile = {percentile}, number of values > percentile = {n_filters}"
            )

    # Boolean mask to filter out the selected filters
    filter_selection = np.concatenate(filter_selection)

    # Dictionary of the selected filters
    filter_selection_dictlist = list(np.array(pyramid.view.filters)[filter_selection])
    
    if verbose:
        print(
            f"Filter includes {np.sum(filter_selection)} out of {pyramid.view.nfilters} filters"
        )
    if plot:
        plot_filter_locations(
            gabor_pyramid=pyramid, in_range=filter_selection, pixdims=(425, 425)
        )
        
    return output_norm, filters_per_freq_sel, filter_selection, filter_selection_dictlist