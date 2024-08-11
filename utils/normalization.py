from enum import Enum
import numpy as np
from types import SimpleNamespace


class Normalization(Enum):
    MINMAX = 1
    ZSCORE = 2
    MINMAX_ALL = 3


def normalize(function: Normalization,
              data: np.ndarray,
              norm_params: SimpleNamespace,
              data_axis: int = None
              ):
    """ Takes the provided data and normalize it using the min and the max values of the data. If the min and max
        is already provided, the stored max/min value will be used.

    :param function: (Normalization) Desired Normalization method
    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param data_axis: (int) Determines which axis should be considered to be the data axis and
     which should be the feature axis

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
    if norm_params.feature_mean_vector is None or not len(norm_params.feature_mean_vector):
        feature_mean_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_mean_vector.append(np.mean(data[:, iFeat]))
        norm_params.feature_mean_vector = feature_mean_vector

    # Extract variance per feature
    if norm_params.feature_var_vector is None or not len(norm_params.feature_var_vector):
        feature_var_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            d = data[:, iFeat]
            feature_var_vector.append(np.var(d))
        norm_params.feature_var_vector = feature_var_vector

    # Extract min values per feature
    if norm_params.feature_min_vector is None or not len(norm_params.feature_min_vector):
        norm_params.data_min = np.min(np.min(data))
        feature_min_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_min_vector.append(np.min(data[:, iFeat]))
        norm_params.feature_min_vector = feature_min_vector

    # Extract max values per feature
    if norm_params.feature_max_vector is None or not len(norm_params.feature_max_vector):
        norm_params.data_max = np.max(np.max(data))
        feature_max_vector = []
        for iFeat in range(0, data.shape[feat_axis]):
            feature_max_vector.append(np.max(data[:, iFeat]))
        norm_params.feature_max_vector = feature_max_vector

    # --- Normalize ---
    if function == Normalization.MINMAX:
        data = norm_minmax(data, norm_params, feat_axis)
    elif function == Normalization.ZSCORE:
        data = norm_zscore(data, norm_params, feat_axis)
    elif function == Normalization.MINMAX_ALL:
        data = norm_minmax_all(data, norm_params, feat_axis)
    else:
        raise ValueError

    return data


def denormalize(function: Normalization,
                data: np.ndarray,
                norm_params: SimpleNamespace,
                ):
    """ Takes the provided data and un-normalizes it using the give de-normalization method

    :param function: (Normalization) Desired De-normalization method
    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :return data: (npArray) Un-Normalized data
    """
    data = np.array(data)
    data = data.astype('float64')

    # Set data and feature axis
    if data.shape[0] > data.shape[1]:
        feat_axis = 1
    else:
        feat_axis = 0

    if function == Normalization.MINMAX:
        data = denorm_minmax(data, norm_params)
    elif function == Normalization.ZSCORE:
        data = denorm_zscore(data, norm_params, feat_axis)
    elif function == Normalization.MINMAX_ALL:
        data = denorm_minmax_all(data, norm_params, feat_axis)
    else:
        raise ValueError
    return data


def norm_minmax(data: np.ndarray, norm_params: SimpleNamespace, feat_axis: int = 0):
    """ Normalizes every feature to the range of -1 to 1 by using the mean and max per feature

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param feat_axis (int) Axis to be considered the feature axis, other axis will be the data axis.

    return data (npArray) Normalized data
    """
    for iFeat in range(0, data.shape[feat_axis]):
        d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
        d_norm = ((2 * (d - norm_params.feature_min_vector[iFeat])) /
                  (norm_params.feature_max_vector[iFeat] - norm_params.feature_min_vector[iFeat]) - 1)

        if feat_axis:
            data[:, iFeat] = d_norm
        else:
            data[iFeat, :] = d_norm
    return data


def denorm_minmax(data: np.ndarray, norm_params: SimpleNamespace):
    """ De-normalizes every feature from the range of -1 to 1 by using the mean and max per feature

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data

    return data (npArray) Denormalized data
    """
    # ----- Reverse Normalize data -----
    for iData in range(0, data.shape[0]):
        d = np.array(data[iData, :])
        d_unnorm = (((d + 1) * (norm_params.data_max - norm_params.data_min)) / 2 + norm_params.data_min)
        data[iData, :] = d_unnorm
    return data


def norm_zscore(data: np.ndarray, norm_params: SimpleNamespace, feat_axis: int = 0):
    """ Normalizes the given data using the Z-score method

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param feat_axis (int) Axis to be considered the feature axis, other axis will be the data axis.

    return data (npArray) Normalized data
    """
    for iFeat in range(0, data.shape[feat_axis]):
        d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
        d_norm = (d - norm_params.feature_mean_vector[iFeat]) / norm_params.feature_var_vector[iFeat]

        if feat_axis:
            data[:, iFeat] = d_norm
        else:
            data[iFeat, :] = d_norm
    return data


def denorm_zscore(data: np.ndarray, norm_params: SimpleNamespace, feat_axis: int = 0):
    """ De-normalizes the given data using the Z-score method

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param feat_axis (int) Axis to be considered the feature axis, other axis will be the data axis.

    return data (npArray) Denormalized data
    """
    for iFeat in range(0, data.shape[feat_axis]):
        d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
        d_unnorm = d * norm_params.feature_var_vector[iFeat] + norm_params.feature_mean_vector[iFeat]
        if feat_axis:
            data[:, iFeat] = d_unnorm
        else:
            data[iFeat, :] = d_unnorm
    return data


def norm_minmax_all(data: np.ndarray, norm_params: SimpleNamespace, feat_axis: int = 0):
    """ Normalizes every feature to the range of -1 to 1 by using the mean and max for all features

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param feat_axis (int) Axis to be considered the feature axis, other axis will be the data axis.

    return data (npArray) Normalized data
    """
    for iFeat in range(0, data.shape[0]):
        d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
        d_norm = ((2 * (d - norm_params.data_min)) / (norm_params.data_max - norm_params.data_min) - 1)

        if feat_axis:
            data[:, iFeat] = d_norm
        else:
            data[iFeat, :] = d_norm
    return data


def denorm_minmax_all(data: np.ndarray, norm_params: SimpleNamespace, feat_axis: int = 0):
    """ De-normalizes every feature from the range of -1 to 1 by using the mean and max for all features

    :param data:  (npArray) Numpy array corresponding to the data to be used within the neural network
    :param norm_params (SimpleNamespace)
                feature_mean_vector=[],  # (list) average values in data per feature
                feature_var_vector=[],   # (list) variance values in data per feature
                feature_min_vector=[],   # (list) min values in data per feature
                feature_max_vector=[],   # (list) max values in data per feature
                data_min=[],             # (float) min value in input data
                data_max=[],             # (float) max value in input data
    :param feat_axis (int) Axis to be considered the feature axis, other axis will be the data axis.

    return data (npArray) Denormalized data
    """
    for iFeat in range(0, data.shape[0]):
        d = np.array(data[:, iFeat]) if feat_axis else np.array(data[iFeat, :])
        d_unorm = ((d + 1) * (norm_params.data_max - norm_params.data_min)) / 2 + norm_params.data_min
        if feat_axis:
            data[:, iFeat] = d_unorm
        else:
            data[iFeat, :] = d_unorm
    return data
