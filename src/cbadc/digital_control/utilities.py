"""Convenience functions for digital control design."""
import numpy as np
import itertools
import scipy.optimize


def overcomplete_set(Gamma: np.ndarray, M: int):
    """
    Construct a overcomplete set of normalized column vectors

    Parameters
    ----------
    Gamma: array_like
        the initial set of vectors
    M : `int`
        the desired number of column vectors

    Returns
    -------
    array_like
        the resulting set of column vectors.
    """
    T = np.copy(Gamma.transpose())
    for dim in range(T.shape[0]):
        T[dim, :] /= np.linalg.norm(T[dim, :], ord=2)
    number_of_candidates_per_new_vector = 100
    while T.shape[0] < M:
        candidate_set = np.random.randn(T.shape[1], number_of_candidates_per_new_vector)
        candidate_set /= np.linalg.norm(candidate_set, ord=2, axis=0)

        cost = np.zeros(number_of_candidates_per_new_vector)

        def cost_function(alpha):
            return np.linalg.norm(np.dot(T, alpha), ord=2) / np.linalg.norm(
                alpha, ord=2
            )

        for index in range(number_of_candidates_per_new_vector):
            sol = scipy.optimize.minimize(cost_function, candidate_set[:, index])
            cost[index] = sol.fun
            candidate_set[:, index] = sol.x / np.linalg.norm(sol.x, ord=2)

        best_candidate_index = np.argmax(cost)

        T = np.vstack(
            (T, candidate_set[:, best_candidate_index].reshape((1, T.shape[1])))
        )
    return T.transpose()


def unit_element_set(N: int, M: int, candidates=[-1, 1, 0]):
    """
    Construct an overcomplete set of vectors only using a single element, i.e.,

    :math:`\mathbf{v} \in \\{ - \\alpha, \\alpha , 0 \\}^{N \\times M}`

    where duplicates and the :math:`\\begin{pmatrix}0, \dots, 0 \\end{pmatrix}`
    is excluded from the set.

    Parameters
    ----------
    N: `int`
        the length of the vectors
    M: `int`
        the number of unique vectors
    candidates: `list[int]`, `optional`
        candidates to permute, defaults to [-1, 1, 0].

    Returns
    -------
    array_like, shape=(N, M)
        a matrix containing the unique vectors as column vectors.
    """
    candidate_set = []
    for item in itertools.product(*[candidates for _ in range(N)]):
        duplicate = False
        sum = np.sum(np.abs(np.array(item)))
        if sum == 0:
            break
        candidate = np.array(item)
        for item in candidate_set:
            s1 = np.sum(np.abs(np.array(item) - candidate))
            s2 = np.sum(np.abs(np.array(item) + candidate))
            if s1 == 0 or s2 == 0:
                duplicate = True
        if not duplicate:
            candidate_set.append(candidate)

    candidate_set = np.array(candidate_set)  # [
    #        np.random.permutation(len(candidate_set)), :
    # ]
    if candidate_set.shape[0] < M:
        raise Exception("Not enough unique combinations; M is set to large.")
    set = candidate_set[0, :].reshape((N, 1))
    candidate_set = np.delete(candidate_set, 0, 0)

    while set.shape[1] < M:
        costs = np.linalg.norm(
            np.dot(candidate_set, set), ord=2, axis=1
        ) / np.linalg.norm(candidate_set, axis=1, ord=2)
        next_index = np.argmin(costs)
        set = np.hstack((set, candidate_set[next_index, :].reshape((N, 1))))
        candidate_set = np.delete(candidate_set, next_index, 0)
    # return np.array(set)[:, np.random.permutation(set.shape[1])]
    return np.array(set)
