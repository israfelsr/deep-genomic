import numpy as np


def drop_unnamed(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def dummy_dataset(inputs, condition, n=8):
    mask = np.random.randint(0, len(inputs), n)
    inputs = inputs[mask]
    condition = condition[mask]
    return inputs, condition


def haploidization(x):
    # heterozygous loci
    n, m = x.shape
    x = x.flatten()
    idx = np.where(x == 1)
    new_values = np.random.randint(2, size=len(idx[0]))
    x[idx[0]] = new_values
    # homozygous loci
    idx = np.where(x == 2)
    x[idx[0]] = 1
    return x.reshape(n, m)


def filter_by(array_to_filter, filter):
    filtered_array = []
    for i in np.unique(filter):
        filtered = array_to_filter[filter == i]
        filtered_array.append(np.sum(filtered, axis=0) / len(filtered))
    return np.asarray(filtered_array)


def normalization(c):
    c_std = c - np.mean(c, axis=0)
    c_std /= np.std(c_std, axis=0)
    return c_std


def split_dataset(x, c, test_percentage=0.2):
    n = x.shape[0]
    len_test_split = int(n * test_percentage)
    idx = np.random.choice(range(n), size=(len_test_split,), replace=False)
    mask = np.full(n, False, dtype=bool)
    mask[idx] = True
    x_test = x[mask]
    x_train = x[~mask]
    c_test = c[mask]
    c_train = c[~mask]
    return x_test, x_train, c_test, c_train