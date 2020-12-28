import pytest
import os
from astrowaves.tasks.NeighbourFinder import find_neighbors

# hardcore_example_path = 'examples/Cont_AA_2_1_small_copy'  # r'examples/A_1_3_adj'
hardcore_example_path = 'examples/A_1_3_adj_copy'

input_path1 = os.path.join(hardcore_example_path, '')


# def test_finding_neighbours():
#     find_neighbors(hardcore_example_path, hardcore_example_path)
#     assert os.path.exists(os.path.join(hardcore_example_path, 'neighbors.csv'))

@pytest.fixture
def input_path():
    return input_path1


@pytest.fixture
def output_path():
    return input_path1


def test_neighbors_finding(input_path, output_path):
    find_neighbors(input_path, output_path)
    assert os.path.exists(os.path.join(output_path, 'neighbors.csv'))
    assert os.path.exists(os.path.join(output_path, 'neighbors_statistics.csv'))


if __name__ == '__main__':
    pytest.main()
