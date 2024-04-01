import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Utility function to visualize dictionary structures
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
            
            
def print_large(item):
    with np.printoptions(threshold=np.inf):
        print(item)
        
        
def get_zscore(data, print_ars = 'y'):
    mean_value = np.mean(data)
    std_dev = np.std(data)

    # Calculate z-scores
    z_scores = (data - mean_value) / std_dev

    if print_ars == 'y':
        print("Original array:", data)
        print("Z-scores:", z_scores)
        
    return z_scores

def mean_center(data, print_ars = 'y'):
    mean_value = np.mean(data)

    # Mean centering
    centered_data = data - mean_value

    if print_ars == 'y':
        print("Original array:", data)
        print("Centered data:", centered_data)
        
    return centered_data
 
def multiple_regression(X, y):
    # Add a column of ones to the end of the X matrix
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

# Function to generate a bell-shaped vector
def generate_bell_vector(n, width, location, kurtosis=0, plot = 'y'):
    x = np.linspace(0, 1, n)
    y = np.exp(-0.5 * ((x - location) / width) ** 2)
    
    if kurtosis != 0:
        y = y ** kurtosis
    
    y /= np.sum(y)  # Normalize the vector to sum up to 1
    
    if plot == 'y':
        plt.scatter(x, y)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Bell-Shaped Vector')
        plt.show()
        
    return y

# Function to plot the hypotheses for the feature and unpredictability sensitivity
def hypotheses_plot():

    hypothesis_1 = hypothesis_2 = np.zeros((4, 5))
    for i in range(4):
        loc = i/3
        # generate_bell_vector(5, 0.15, loc)
        hypothesis_1[i, :] = np.array(generate_bell_vector(5, 0.0115, loc, 0.01, plot = 'n'))
        
    hypothesis_3 = np.zeros((4, 5))
    for i in range(4):
        loc = i/3
        # generate_bell_vector(5, 0.15, loc)
        hypothesis_3[np.abs(i - 3), :] = np.array(generate_bell_vector(5, 0.0115, loc, 0.01, plot = 'n'))
        
    # Keeping the same values for Hypothesis 4
    hypothesis_4 = np.zeros((4, 5))
    for i in range(4):
        hypothesis_4[i, :] = hypothesis_1[3]

    hypotheses = [hypothesis_1, hypothesis_2, hypothesis_3, hypothesis_4]

    # Visual areas
    visual_areas = ['V1', 'V2', 'V3', 'V4']

    # Define a gradient colormap from dark blue to light red
    cmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000080', '#CCCCCC', '#FFA500', '#FF0000'], N=5)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(np.array([14, 4.25])*1.1), sharey=True)
        
    plt.subplots_adjust(wspace=-1)  # Adjust this value to your liking

    for i, hypothesis in enumerate(hypotheses):
        ax = axs[i]
        # Transpose the hypothesis so that each vector becomes a column
        hypothesis_t = np.transpose(hypothesis)
        bottom = np.zeros(len(hypothesis_t[0]))
        for j, data in enumerate(hypothesis_t):
            ax.bar(range(len(data)), data, bottom=bottom, edgecolor='none', linewidth=0.5, label=f'{j+1}', color=cmap(j))
            bottom += data
        ax.set_title(['Visual feature sensitivity\n\n\n', 
                    'Hypothesis 1: \nTraditionally hierarchical\nunpredictability sensitivity\n', 
                    'Hypothesis 2: \nReversed hierarchical\nunpredictability sensitivity\n', 
                    'Hypothesis 3: \nNon-hierarchical\nunpredictability sensitivity\n'][i], 
                    fontweight='normal', family = 'sans-serif', fontsize = 16)
        ax.set_ylim(0, 1)  # Adjust the y-axis limits based on your data range
        ax.set_yticks([0, 0.5, 1])  # Set y-ticks
        ax.set_yticklabels([0, 50, 100])  # Change y-tick labels
        ax.spines['top'].set_visible(False)  # Remove top border
        ax.spines['right'].set_visible(False)  # Remove right border
        ax.set_xticks(range(len(visual_areas)))  # Set x-ticks
        ax.set_xticklabels(visual_areas)  # Set x-tick labels
        ax.grid(False)  # Remove grid

    # Remove 'Category' label from the bottom
    fig.text(0.5, 0, '', ha='center', va='center', fontsize=14)
    fig.text(0.0, 0.43, 'Layer Assignment (%)', va='center', rotation='vertical', fontweight='normal', fontsize = 15)
    fig.text(0.105, -.01, 'Visual Areas', ha='left', fontweight='normal', fontsize = 15)

    axs[0].legend(title='CNN Layer', loc = 'upper center', bbox_to_anchor=(0.5, 1.27),
            ncol=5, fancybox=False, shadow=False, fontsize = 11.5, columnspacing = .55)

    plt.tight_layout()
    plt.show()
    
    
# Turn a boolean nifti mask  or 3d
def numpy2coords(boolean_array):
    # Get the coordinates of the True values in the boolean array
    coordinates = np.array(np.where(boolean_array)).T
    return coordinates

def coords2numpy(coordinates, shape):
    # Create a boolean array with the same shape as the original array
    boolean_array = np.zeros(shape, dtype=bool)
    # Set the coordinates to True
    boolean_array[tuple(coordinates.T)] = True
    return boolean_array

# Function to return the voxel coordinates based on the parameter represented in the 4th column
def filter_array_by_size(array, size_min, size_max):
    filtered_array = array[(array[:, 3] >= size_min) & (array[:, 3] <= size_max)]
    return filtered_array

# give new array that only contains the common rows, used for post hoc voxel selection.
def find_common_rows(array1, array2):
    set1 = set(map(tuple, array1[:,:3]))
    set2 = set(map(tuple, array2[:,:3]))
    common_rows = np.array([list(x) for x in set1 & set2])
    return common_rows

def ecc_angle_to_coords(ecc, angle, dim = 425):
    
    y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
    x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
    
    x = ecc * np.cos(np.radians(angle))
    y = ecc * np.sin(np.radians(angle))
    return x, y
