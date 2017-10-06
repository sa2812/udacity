#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    num_to_remove = len(predictions)/10

    diff = np.square(net_worths - predictions)

    for ii in range(num_to_remove):
        highest_index = np.argmax(diff)

        ages = np.delete(ages, highest_index)
        net_worths = np.delete(net_worths, highest_index)
        diff = np.delete(diff, highest_index)

    cleaned_data = zip(ages, net_worths, diff)
    
    return cleaned_data

