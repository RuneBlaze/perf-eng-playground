from typing import Callable, TypeVar, Iterable
import concurrent.futures

T = TypeVar('T')
U = TypeVar('U')

def thread_map(f: Callable[[T], U], sequence: Iterable[T]) -> Iterable[U]:
    """
    Map function `f` over the elements of `sequence` using 4 threads.

    Parameters:
    - f: The function to apply on each element of the sequence.
    - sequence: The input sequence.

    Returns:
    - A list with the results of applying `f` on each element of `sequence`.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return executor.map(f, sequence)