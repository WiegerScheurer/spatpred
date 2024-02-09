import numpy as np
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