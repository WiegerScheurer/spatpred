# Utility function to visualize dictionary structures
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
