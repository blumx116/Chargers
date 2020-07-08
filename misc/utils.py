import itertools
import os
from pathlib import Path
from typing import List, Iterable, TypeVar, Union, Tuple, Generic

import numpy as np
from numpy.random import RandomState
import torch

T = TypeVar("T")
V = TypeVar("V")

def flatmap(data: Iterable[Iterable[T]]) -> List[T]:
    return [el for l in data for el in l]

def df_flatmap(df, func):
    return list(itertools.chain.from_iterable(
        [func(row[1]) for row in df.iterrows()]))

def unpack_tuple(tup: Tuple[T]) -> Union[T, Tuple[T]]:
    if not isinstance(tup, tuple):
        return tup #only unpack tuples
    elif len(tup) > 1:
        return tup
    elif len(tup) == 1:
        return tup[0]
    else: #len == 0
        return None

def listify(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def optional_random(rand_seed: Union[int, RandomState] = None):
    if rand_seed is None:
        return np.random
    elif isinstance(rand_seed, int):
        return RandomState(rand_seed)
    else:
        return rand_seed


def array_unique(data: Iterable[np.ndarray], return_inverse=False, 
        return_counts = False) -> List[np.ndarray]:
    """
        used for essentially calling np.unique on a list of np.ndarrays
        NOTE: this is only needed because the default np.unique function can't handle arrays of arrays
        Parameters
        ----------
        data: Iterable[np.ndarray] : n_data => [ndarray_dims]
            array of np.ndarrays to find unique elements from
        return_inverse: bool
            see np.unique docs
        return_counts: bool
            see np.unique docs
        Returns
        -------
        uniquas : List[np.ndarray] : n_unique => [ndarray_dims]
            if return_inverse : np.ndarray[int] : [n_data] # see np.unique docs
            if return_counts  : np.ndarray[int] : [n_unique] # ditto
    """
    data : np.ndarray = np.vstack(data) # [n_data, ndarray_dims]
    filtered, inv_ret, count_ret = np.unique(data, axis=0, return_inverse=True, return_counts=True) 
    # np.ndarray : [n_unique, ndarray_dims], np.ndarray : [n_unique,], np.ndarray : [n_data,]
    result = [filtered[i] for i in range(filtered.shape[0])]
    # result: List[np.ndarray] : n_unique => [ndarray_dims]
    result = (result, )
    if return_inverse:
        result = result + (inv_ret, )
    if return_counts:
        result = result + (count_ret, )
    return unpack_tuple(result)


def array_equal(el1: T, el2: T) -> bool:
    """
        Checks if element 1 and 2 are equal
        NOTE: this only exists because numpy's default handling of the == operator
        is different from that of normal objects. This provides a data agnostic way
        to check
        Parameters
        ----------
        el1: T
            left hand side of equality
        el2: T
            right hand side of equality
        Returns
        -------
        equal?: bool
            whether or not el1 == el2
    """
    if isinstance(el1, np.ndarray) and isinstance(el2, np.ndarray):
        return np.array_equal(el1, el2)
    elif isinstance(el1, Iterable) and isinstance(el2, Iterable):
        if len(el1) != len(el2):
            return False 
        if len(el1) == len(el2) == 0:
            return True 
        else:
            return array_equal(el1[0], el2[0]) and array_equal(el1[1:], el2[1:])
    elif type(el1) != type(el2):
        return False
    else:
        return el1 == el2


def array_contains(el: T, elems: Iterable[T]) -> bool:
    """
        returns whether or not el is in elems
        NOTE: This only exists because the default contains method doesn't work with np.ndarrays
        Parameters
        ----------
        el: T
            element to check for in elems
        elems: Iterable[T]
            elements to check whether el is a part of
        Returns
        -------
        present: bool
            whether or not el was present in elems
        
    """
    for element in elems:
        if array_equal(el, element):
            return True
    return False


def array_random_choice(
        elems: Iterable[T],
        probas: Iterable[float] = None,
        random: Union[int, RandomState] = None,
        count: int = 1) -> Union[T, List[T]]:
    """
        Randomly selects an element from elems to return. 
        NOTE: Exists because default numpy function not compatible with np.ndarrays
        Parameters
        ----------
        elems : Iterable[T]]
            elements to choose from among
        probas: Iterable[float]: [len(elems), ]
            probabilities of choosing each item - need not sum to 1
        random : Union[int, RandomState]
            random seed to use, uses global random if none provided
        count: int > 0 
            number of samples to return. If >1, will pack in to a list
        Returns
        -------
        selected: T
            the randomly selected element from the list
            if count > 1, packs results in to a list
    """
    assert count > 0
    random: RandomState = optional_random(random)
    elems: List[T] = list(elems)
    if probas is not None:
        probas: np.ndarray = np.asarray(probas, dtype=float) + 1e-8
        probas /= np.sum(probas)
    random: RandomState = optional_random(random)
    def sample():
        idx: int = random.choice(len(elems), p=probas)
        return elems[idx]
    if count == 1:
        return sample()
    else:
        return [sample() for _ in range(count)]


def array_shuffle(elems: List[np.ndarray], random: Union[int, RandomState] = None) -> List[np.ndarray]:
    """
        shuffles an array of arrays, out of place
        NOTE: Exists because default numpy function not compatible with np.ndarrays
        elems: array of np.ndarrays to shuffle
        Parameters
        ----------
        elems: List[np.ndarray]
            elements to be shuffled, not in place. Technically works on even if 
            elements are not np.ndarrays
        random: Union[int, RandomState]
            random seed to use, uses global random if none provided
        Returns
        -------
        shuffled: List[np.ndarray]
            list with same elements as elems, in shuffled order
    """
    random = optional_random(random)
    elems = list(elems)
    indices = np.arange(len(elems))
    random.shuffle(indices)
    return [elems[i] for i in indices]


def bool_random_choice(probability: float, rand_seed: Union[int, RandomState] = None) -> bool:
    """
        Randomly chooses True with probability equal to 'probability'
        and False otherwise, using rand_seed for random number generator
        Possibly redundant.
        Parameters
        ----------
        probability: float in [0, 1]
            probability of returning True
        rand_seed: Union[int, RandomState]
            random seed to use, uses global random if none provided
        Returns
        -------
        choice: bool
            True or False, chosen randomly
        """
    assert 0 <= probability <= 1
    random: RandomState = optional_random(rand_seed)
    return random.uniform() < probability

PROJECT_BASE_DIR: str = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir))


def is_onehot(vec: Union[torch.Tensor, np.ndarray]) -> bool:
    return vec.sum() == vec.max() == 1.


def np_onehot(values: Union[int, List[int], np.ndarray], max: int) -> np.ndarray:
    if isinstance(values, int):
        value: int = values
        assert 0 <= value <= max
        return (np.arange(max+1) == value).astype(int)
    values = np.asarray(values)
    assert len(values.shape) <= 2
    if len(values.shape) == 1:
        #credit: https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
        return np.eye(max+1)[values].astype(int)
    else:
        # credit: https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
        return (np.arange(max+1) == values[..., None]).astype(int)

root_dir: Path = Path(__file__).parent.parent.absolute()