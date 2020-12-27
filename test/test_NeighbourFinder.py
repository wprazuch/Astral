import pytest
import os
from astrowaves.tasks.NeighbourFinder import find_neighbors

hardcore_example_path = r'examples/A_1_3_adj'

input_path = os.path.join(hardcore_example_path, '')


def test_finding_neighbours():
    find_neighbors(hardcore_example_path, hardcore_example_path)
    assert os.path.exists(os.path.join(hardcore_example_path, 'neighbors.csv'))


if __name__ == '__main__':
    pytest.main()
