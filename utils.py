import time
import numpy as np
import torch
import matplotlib.pyplot as plt

##numpy utils

def print_np_array_properties(np_array, name='np array'):
    '''This Function prints the properties of an NumPy Array'''
    print(f'=============== {name} Properties =============\n')
    print(f'{name}:\n {np_array}\n{name}_dtype: {np_array.dtype}\n{name}_shape: {np_array.shape}\n{name}_size: {np_array.size}\n')
    print('================================================')

## Data processing Helper functions

def split_validation(data):
    
    valid_len = len(data)//2
    test_len = len(data) - valid_len
    valid_data, test_data = torch.utils.data.random_split(data, [valid_len , test_len])
    
    return valid_data, test_data

## Encoded labels helper functions
def get_encoded_attrs(classes, label):
	
	attr = [classes[idx] for idx, attr in enumerate(label) if attr]
	
	return attr


def encode_attrs(attrs, classes):
	
	endoded_label = [1 if label in attrs else 0 for label in classes]
	
	return encoded_label
	

## Displaying output helper functions

def imshow_with_encoded_labels(batch_size, images, labels, classes):
	
	for i in np.arange(batch_size):
    
		fig = plt.figure(num='w/attrs')

		ax = fig.add_subplot(xticks=[], yticks=[])
		plt.imshow(images[i].permute(1, 2, 0) if torch.is_tensor(images) else np.transpose(images[i], (1, 2, 0)) if isinstance(images, np.ndarray) else None)
		
		attr=get_encoded_attrs(classes, labels[i])

		ax.set_title(attr)

def imshow_batch(batch_size, images):
	
	ncols = 10
	nrows = batch_size // ncols + (batch_size % ncols > 0)
	
	fig = plt.figure( num='batched', figsize=(25,5*nrows), layout='tight')

	for i in np.arange(batch_size):

		ax = plt.subplot(nrows, ncols, i + 1, xticks=[], yticks=[])
		# ax.set_title(z[i])
		plt.imshow(images[i].permute(1, 2, 0) if torch.is_tensor(images) else np.transpose(images[i], (1, 2, 0)) if isinstance(images, np.ndarray) else None)

def imshow_compared_reconstuction():
	pass
