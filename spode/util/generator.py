import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
from math import floor

__all__ = ['get_supported_placement', 'generate']

# stores all the supported placements for automatic generation
_placement_size_dict = {'square_1': [2, 'A basic square mesh of size N by M. Two integers are needed to define this '
                                     'placement, representing #Row and #Col.'],
                     'hexagonal_1': [1, 'A concentric hexagonal mesh with N levels. One integer is needed to define '
                                        'this placement, representing #level.'],
                     'triangular_1': [2, 'A triangular mesh structure of size N by M (close to diamond). Two integers '
                                         'are needed to define this placement, representing #Row and #Col.']}


def get_supported_placement():
    """Print the supported placements."""

    print("\nAutomatic schematic generation is supported for the following placements:")
    for k, v in _placement_size_dict.items():
        print("### config = '%s', len(size) = %d. %s" % (k, v[0], v[1]))
    print("\n")


def generate(placement: str, size: List[int], init_dict: dict, tbu_model: Optional[str] = 'tbum'):
    """Generate the circuit netlist

    Generate the circuit netlist based on placement, size, and tbu_model.

    :param placement: A string specify the placement, e.g., 'square_1'.
    :param size: A list of integer specify the size of the programmable photonic circuit.
                 The length of it depends on the variable 'config'.1
    :param init_dict: specify the initial values for all TBUs.
    :param tbu_model: specify which tbu model to use, e.g., 'tbum', 'tbuo', 'tbut'.
    :return: a dict contains all circuit elements.
    """
    return Generator().generate(placement, size, init_dict, tbu_model)


class Generator(object):
    """Automatic programmable photonics circuit schematic generator"""

    def __init__(self):
        pass

    def generate(self, placement: str, size: List[int], init_dict: dict, tbu_model: Optional[str] = 'tbum') -> dict:
        """Generate the circuit netlist

        Generate the circuit netlist based on self.placement, self.size, and self.tbu_model.

        :param placement: A string specify the placement, e.g., 'square_1'.
        :param size: A list of integer specify the size of the programmable photonic circuit.
                     The length of it depends on the variable 'config'.1
        :param init_dict: specify the initial values for all TBUs.
        :param tbu_model: specify which tbu model to use, e.g., 'tbum', 'tbuo', 'tbut'.
        :return: a dict contains all circuit elements.
        """

        self.placement = placement
        self.size = size
        self.tbu_model = tbu_model

        if self.placement not in _placement_size_dict.keys():
            raise RuntimeError("The provided placement is not supported for automatic schematic generation.")
        elif _placement_size_dict[self.placement][0] != len(self.size):
            raise RuntimeError("%d integer(s) required to define this placement, but %d is provided."
                               % (_placement_size_dict[self.placement][0], len(self.size)))
        else:
            return getattr(self, '_generate_' + self.placement)(init_dict)

    def _generate_square_1(self, init_dict) -> dict:
        """Generate circuit elements for a N by M square mesh (will be called in generate function).

        :param init_dict: specify the initial values for all TBUs
        :return" a dict contains all circuit elements.

        """
        circuit_element = {}

        # our 'ln' and 'rn' follow:
        #     -------theta
        #     -------phi
        # |  |
        # |  |
        # phi theta

        # process all vertical TBUs
        for i in range(1, self.size[0] + 1):
            for j in range(0, self.size[1] + 1):
                cur_tbu_model = self.tbu_model + '_' + str(i) + '#' + str(j + 1) + '_' + str(i) + '#' + str(j) + '_v'
                node_dict = {
                    'ln': ['n' + '_' + str(i) + '#' + str(j + 1) + '_' + 'bl',
                           'n' + '_' + str(i) + '#' + str(j) + '_' + 'br'],
                    'rn': ['n' + '_' + str(i) + '#' + str(j + 1) + '_' + 'tl',
                           'n' + '_' + str(i) + '#' + str(j) + '_' + 'tr']}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        # process all horizontal TBUs
        for i in range(0, self.size[0] + 1):
            for j in range(1, self.size[1] + 1):
                cur_tbu_model = self.tbu_model + '_' + str(i) + '#' + str(j) + '_' + str(i + 1) + '#' + str(j) + '_h'
                node_dict = {
                    'ln': ['n' + '_' + str(i) + '#' + str(j) + '_' + 'br',
                           'n' + '_' + str(i + 1) + '#' + str(j) + '_' + 'tr'],
                    'rn': ['n' + '_' + str(i) + '#' + str(j) + '_' + 'bl',
                           'n' + '_' + str(i + 1) + '#' + str(j) + '_' + 'tl']}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        return circuit_element

    def _generate_hexagonal_1(self, init_dict) -> dict:
        """Generate circuit elements for a concentric hexagonal mesh (will be called in generate function).

        :param init_dict: specify the initial values for all TBUs
        :return" a dict contains all circuit elements.

        """

        def _helper_mod(dividend: int, divisor: int) -> int:
            """A helper modulus function

            For example, if divisor = 4, then:
            it returns 1, given dividend = 1
            it returns 2, given dividend = 2
            it returns 3, given dividend = 3
            it returns 4, given dividend = 4
            it returns 1, given dividend = 5
            """
            return (dividend - 1) % divisor + 1

        # level = 1,2,3,...
        num_tbu_func = lambda level: max(6 * (level - 1), 6)
        num_radial_edge_func = lambda level: 6 * (level - 1)
        num_normal_edge_func = lambda level: 12 * level - 6

        circuit_element = {}

        magic_connect = [{'ln': ['p1', 'p3'], 'rn': ['p6', 'p4']},
                         {'ln': ['p2', 'p4'], 'rn': ['p1', 'p5']},
                         {'ln': ['p3', 'p5'], 'rn': ['p2', 'p6']},
                         {'ln': ['p4', 'p6'], 'rn': ['p3', 'p1']},
                         {'ln': ['p5', 'p1'], 'rn': ['p4', 'p2']},
                         {'ln': ['p6', 'p2'], 'rn': ['p5', 'p3']}, ]

        magic_index = np.array([4, 5, 0, 1, 2, 3], dtype=np.int)
        index_add = magic_index + 1

        num_layer = self.size[0]

        for i in range(1, num_layer + 1):
            num_tbus = num_tbu_func(i)
            num_radial_edges = num_radial_edge_func(i)

            # deal with radial edges
            for j in range(1, num_radial_edges + 1):
                # map {1,2,...,num_radial_edge} to {2,3,..., 1}
                ind_j = j % num_radial_edges + 1

                cur_tbu_model = self.tbu_model + '_' + str(i) + '#' + str(j) + '_' + str(i) + '#' + str(ind_j) + '_rd'

                indicator = (j - 1) // (i - 1)  # which one of the magic_connect
                ln = ['n' + '_' + str(i) + '#' + str(j) + '_' + magic_connect[indicator]['ln'][1],
                      'n' + '_' + str(i) + '#' + str(ind_j) + '_' + magic_connect[indicator]['ln'][0]]
                rn = ['n' + '_' + str(i) + '#' + str(j) + '_' + magic_connect[indicator]['rn'][1],
                      'n' + '_' + str(i) + '#' + str(ind_j) + '_' + magic_connect[indicator]['rn'][0]]

                node_dict = {'ln': ln, 'rn': rn}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

            # deal with normal edges
            start_ind_list = magic_index * (i - 1) + 1
            end_ind_list = (magic_index + 2) * (i - 1) + 1
            for m in range(len(start_ind_list)):
                for k in range(start_ind_list[m], end_ind_list[m] + 1):
                    ind_k = _helper_mod(k, num_tbus)
                    ind_k_next_level = _helper_mod(k + index_add[m], num_tbu_func(i + 1))
                    cur_tbu_model = self.tbu_model + '_' + str(i) + '#' + str(ind_k) + '_' + str(i + 1) + '#' + str(
                        ind_k_next_level) + '_nd'

                    ln = ['n' + '_' + str(i) + '#' + str(ind_k) + '_' + magic_connect[m]['rn'][0],
                          'n' + '_' + str(i + 1) + '#' + str(ind_k_next_level) + '_' + magic_connect[m]['rn'][1]]
                    rn = ['n' + '_' + str(i) + '#' + str(ind_k) + '_' + magic_connect[m]['ln'][0],
                          'n' + '_' + str(i + 1) + '#' + str(ind_k_next_level) + '_' + magic_connect[m]['ln'][1]]


                    node_dict = {'ln': ln, 'rn': rn}
                    circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        return circuit_element

    def _generate_triangular_1(self, init_dict) -> dict:
        """Generate circuit elements for a triangular mesh (will be called in generate function).

                :param init_dict: specify the initial values for all TBUs
                :return" a dict contains all circuit elements.

                """
        magic_connect = [{'ln': ['p1', 'p3'], 'rn': ['p3', 'p1']},
                         {'ln': ['p3', 'p2'], 'rn': ['p2', 'p3']},
                         {'ln': ['p2', 'p1'], 'rn': ['p1', 'p2']}]

        circuit_element = {}
        for i in range(1, self.size[0] + 1):
            # deal with those tilted TBUs
            for j in range(0, self.size[1] + 1):
                cur_tbu_model = self.tbu_model + '_' + str(i) + '#' + str(j) + '_' + str(i) + '#' + str(j + 1) + '_t'
                ln = ['n' + '_' + str(i) + '#' + str(j) + '_' + magic_connect[j % 2]['rn'][0],
                      'n' + '_' + str(i) + '#' + str(j + 1) + '_' + magic_connect[j % 2]['rn'][1]]
                rn = ['n' + '_' + str(i) + '#' + str(j) + '_' + magic_connect[j % 2]['ln'][0],
                      'n' + '_' + str(i) + '#' + str(j + 1) + '_' + magic_connect[j % 2]['ln'][1]]
                node_dict = {'ln': ln, 'rn': rn}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        for i in range(1, self.size[0] + 2):
            # deal with horizontal TBUs
            for j in range(1, floor((self.size[1] - 1) / 2) + 2):
                ind_j = 2 * j - 1
                cur_tbu_model = self.tbu_model + '_' + str(i - 1) + '#' + str(ind_j + 1) + '_' + str(i) + '#' + str(
                    ind_j) + '_h'
                ln = ['n' + '_' + str(i - 1) + '#' + str(ind_j + 1) + '_' + magic_connect[2]['rn'][0],
                      'n' + '_' + str(i) + '#' + str(ind_j) + '_' + magic_connect[2]['rn'][1]]
                rn = ['n' + '_' + str(i - 1) + '#' + str(ind_j + 1) + '_' + magic_connect[2]['ln'][0],
                      'n' + '_' + str(i) + '#' + str(ind_j) + '_' + magic_connect[2]['ln'][1]]
                node_dict = {'ln': ln, 'rn': rn}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}
        return circuit_element


