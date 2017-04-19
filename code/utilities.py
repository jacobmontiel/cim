""" Utility functions
"""

import numpy as np

def convert_obj_to_num(data):
    """ Convert 'object' attributes to 'numerical'
    :param data: DataFrame to processes
    """
    convert = []
    for attr in data.columns:
        if data[attr].dtypes == 'O':
            convert.append(attr)
    if len(convert) > 0:
        print("Found object attributes, will convert them to num:")
        print(convert)
        data.loc[:, convert] = data.loc[:, convert].apply(lambda x: x.astype('category').cat.codes)
        # Replace -1 with NaN
        data.loc[:, convert] = data.loc[:, convert].replace(-1, np.nan)
    return data
