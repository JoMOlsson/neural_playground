from enum import Enum
import numpy as np


class Normalization(Enum):
    MINMAX = 1
    ZSCORE = 2
    MINMAX_ALL = 3
    MINMAX_OLD = 4


def normalize(function: Normalization,
              data: np.ndarray,
              data_axis: int = None,
              feature_mean_vector: list = None,
              feature_var_vector: list = None,
              feature_min_vector: list = None,
              feature_max_vector: list = None,
              data_min=None,
              data_max=None
              ):
    """ Takes the provided data and normalize it using the min and the max values of the data. If the min and max
        is already provided, the stored max/min value will be used.

    # TODO - ASSIGN NAMESPACE INSTEAD OF DICT AND DIRECTLY ASSIGN

    :param function: (Normalization) Desired Normalization method
    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param data_axis: (int) Determines which axis should be considered to be the data axis and
     which should be the feature axis
    :param feature_mean_vector
    :param feature_var_vector
    :param feature_mean_vector
    :param feature_min_vector
    :param feature_max_vector
    :param data_min
    :param data_max
    :return data: (npArray) Normalized data
    """

    data = np.array(data)
    data = data.astype('float64')

    # Set data and feature axis
    if data_axis is None:
        if data.shape[0] > data.shape[1]:
            feat_axis = 1
        else:
            feat_axis = 0
    else:
        feat_axis = 1

    # Assign mean per feature
    if feature_mean_vector is None or not len(feature_mean_vector):
        feature_mean_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_mean_vector.append(np.mean(data[:, iFeat]))

    # Extract variance per feature
    if feature_var_vector is None or not len(feature_var_vector):
        feature_var_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            d = data[:, iFeat]
            feature_var_vector.append(np.var(d))

    # Extract min values per feature
    if feature_min_vector is None or not len(feature_min_vector):
        data_min = np.min(np.min(data))
        feature_min_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_min_vector.append(np.min(data[:, iFeat]))

    # Extract max values per feature
    if feature_max_vector is None or not len(feature_max_vector):
        data_max = np.max(np.max(data))
        feature_max_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_max_vector.append(np.max(data[:, iFeat]))

    # --- Normalize ---
    if function == Normalization.MINMAX:
        for iFeat in range(0, data.shape[feat_axis]):
            d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
            d_norm = ((2 * (d - feature_min_vector[iFeat])) /
                      (feature_max_vector[iFeat] - feature_min_vector[iFeat]) - 1)

            if feat_axis:
                data[:, iFeat] = d_norm
            else:
                data[iFeat, :] = d_norm
    elif function == Normalization.ZSCORE:
        for iFeat in range(0, data.shape[feat_axis]):
            d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
            d_norm = (d - feature_mean_vector[iFeat]) / feature_var_vector[iFeat]

            if feat_axis:
                data[:, iFeat] = d_norm
            else:
                data[iFeat, :] = d_norm
    elif function == Normalization.MINMAX_ALL:
        for iFeat in range(0, data.shape[0]):
            d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
            d_norm = ((2 * (d - data_min)) / (data_max - data_min) - 1)

            if feat_axis:
                data[:, iFeat] = d_norm
            else:
                data[iFeat, :] = d_norm
    elif function == Normalization.MINMAX_OLD:
        for iData in range(0, data.shape[0]):
            d = np.array(data[iData, :])
            d_norm = (2 * (d - data_min)) / (data_max - data_min) - 1
            data[iData, :] = d_norm
    else:
        raise ValueError

    # Pack data
    packed_data = {'data': data, 'feature_mean_vector': feature_mean_vector, 'feature_var_vector': feature_var_vector,
                   'feature_min_vector': feature_min_vector, 'feature_max_vector': feature_max_vector,
                   'data_min': data_min, 'data_max': data_max}
    return packed_data


def denormalize(function: Normalization, data: np.ndarray,
                feature_mean_vector: list = None,
                feature_var_vector: list = None,
                feature_min_vector: list = None,
                feature_max_vector: list = None,
                data_min=None,
                data_max=None
                ):
    """ Takes the provided data and un-normalizes it using the min and the max values of the data.

    TODO - IMPLEMENT FOR ALL NORMALIZATION METHODS
    :param function: (Normalization) Desired De-normalization method
    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param feature_mean_vector
    :param feature_var_vector
    :param feature_mean_vector
    :param feature_min_vector
    :param feature_max_vector
    :param data_min
    :param data_max
    :return data: (npArray) Un-Normalized data
    """
    data = np.array(data)
    data = data.astype('float64')

    if function == Normalization.MINMAX:
        # ----- Reverse Normalize data -----
        for iData in range(0, data.shape[0]):
            d = np.array(data[iData, :])
            d_unnorm = (((d + 1) * (data_max - data_min)) / 2 + data_min)
            data[iData, :] = d_unnorm
    elif function == Normalization.ZSCORE:
        return None
    elif function == Normalization.MINMAX_ALL:
        return None
    elif function == Normalization.MINMAX_OLD:
        return None
    else:
        raise ValueError
    return data
