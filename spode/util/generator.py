import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
from abc import abstractmethod, ABCMeta
import warnings

from spode.core import Circuit

__all__ = ['Generator', 'get_supported_config']

_config_size_dict = {'square_1': [2, 'A basic square mesh of size N by M. Two integers are needed to define this '
                                     'configuration.']}


def get_supported_config():
    count = 1
    print("\nAutomatic schematic generation is supported for the following configurations:")
    for k, v in _config_size_dict.items():
        print("### config = '%s', len(size) = %d. %s" % (k, v[0], v[1]))
    print("\n")


class Generator(object):
    """Automatic programmable photonic circuit schematic generator"""

    def __init__(self, config: str, size: List[int], tbu_model: Optional[str] = 'tbum'):
        """Initialize the Generator according to a input configuration, and an input size.

        :param config: A string specify the configuration, e.g., 'square_1'.
        :param size: A list of integer specify the size of the programmable photonic circuit.
                     The length of it depends on the variable 'config'.1
        :param tbu_model: specify which tbu model to use, e.g., 'tbum', 'tbuo', 'tbut'.
        """

        self.config = config
        self.size = size
        self.tbu_model = tbu_model

    def generate(self, init_dict) -> dict:
        """Generate the circuit netlist

        Generate the circuit netlist based on self.config, self.size, and self.tbu_model.

        :param init_dict: specify the initial values for all TBUs.

        """
        if self.config not in _config_size_dict.keys():
            raise RuntimeError("The provided configuration is not supported for automatic schematic generation.")
        elif _config_size_dict[self.config][0] != len(self.size):
            raise RuntimeError("%d integers are required to define this configuration, but %d is provided."
                               % (_config_size_dict[self.config][0], len(self.size)))
        else:
            return getattr(self, '_generate_' + self.config)(init_dict)

    def _generate_square_1(self, init_dict) -> dict:
        circuit_element = {}
        # process all vertical TBUs
        for i in range(1, self.size[0] + 1):
            for j in range(0, self.size[1] + 1):
                cur_tbu_model = self.tbu_model + '_' + str(i) + str(j) + 'v'
                node_dict = {
                        'ln': ['n' + '_' + str(i) + str(j + 1) + '_' + 'tl', 'n' + '_' + str(i) + str(j) + '_' + 'tr'],
                        'rn': ['n' + '_' + str(i) + str(j + 1) + '_' + 'bl', 'n' + '_' + str(i) + str(j) + '_' + 'br']}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        for i in range(0, self.size[0] + 1):
            for j in range(1, self.size[1] + 1):
                cur_tbu_model = self.tbu_model + '_' + str(i) + str(j) + 'h'
                node_dict = {
                        'ln': ['n' + '_' + str(i) + str(j) + '_' + 'bl', 'n' + '_' + str(i + 1) + str(j) + '_' + 'tl'],
                        'rn': ['n' + '_' + str(i) + str(j) + '_' + 'br', 'n' + '_' + str(i + 1) + str(j) + '_' + 'tr']}
                circuit_element[cur_tbu_model] = {**node_dict, **init_dict}

        return circuit_element


if __name__ == '__main__':
    print("hello worlds")
    a = Generator('square_1', [1, 1])
    circuit_element = a.generate({'theta': 0.0 * np.pi, 'phi': 0.0 * np.pi, 'l': 250e-6})
    print(circuit_element)

    circ_instance = Circuit(circuit_element=circuit_element,
                            mode_info={'neff': 4.0},
                            omega=np.linspace(192.5, 193.5, 3) * 1e12 * 2 * np.pi,
                            srce_node={'n_01_bl':1.0},
                            prob_node=['n_12_bl'],
                            deri_node=[],
                            deri_vari=[])

    response = circ_instance.solve(False)
    print(np.abs(response[0,:,:]))
