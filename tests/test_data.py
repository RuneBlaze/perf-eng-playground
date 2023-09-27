from fib.data import FibDataset

def test_fib_dataset_can_yield_one_dp():
    dataset = FibDataset()
    point = next(iter(dataset))
    assert 'x' in point
    assert 'y' in point