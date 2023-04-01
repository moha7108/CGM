import time
import numpy as np

##numpy utils

def print_np_array_properties(np_array, name='np array'):
    '''This Function prints the properties of an NumPy Array'''
    print(f'=============== {name} Properties =============\n')
    print(f'{name}:\n {np_array}\n{name}_dtype: {np_array.dtype}\n{name}_shape: {np_array.shape}\n{name}_size: {np_array.size}\n')
    print('================================================')
	
	

def split_validation(data):
    
    valid_len = len(data)//2
    test_len = len(data) - valid_len
    valid_data, test_data = torch.utils.data.random_split(data, [valid_len , test_len])
    
    return valid_data, test_data