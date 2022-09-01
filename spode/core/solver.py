import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
import importlib
import json
from functools import reduce
import operator
from copy import deepcopy
from spode.core.model import _model_json_name
from spode.util.visual import _visualize_base, _cell_place_function_list

__all__ = ['Circuit']


# TODO: currently only support constant source, is frequency-dependent source meaningful?

class Circuit(object):
    """The Circuit class object"""

    def __init__(self, circuit_element: dict, mode_info: dict, omega: Union[List[float], np.ndarray],
                 srce_node: dict, prob_node: List[str],
                 deri_node: Optional = None, deri_vari: Optional = None, notation: Optional[str] = ''):
        """Initialize the Circuit class.

        The Circuit object takes a few parameters as input, and then initialize its attributes. In the meanwhile, we will
        also do some preprocessing in the initialization process. The following parameters will be directly stored in the
        Circuit object using the same name.

        :param circuit_element: information about elements and their connections in the concerned circuit.
        :param mode_info: information about the simulating propagating mode, e.g., {'neff':2.0}.
        :param omega: the simulation angular frequency grid points.
        :param srce_node: the node where the sources are injected and the values of sources.
        :param probe_node: specify which ports' complex magnitudes are of interests.
        :param deri_node: specify which port magnitude derivatives are of interests.


        These attributes will be calculated based on the above given parameters. Moreover, in the calculation, we will do some checking about the validity of the input parameters, in case of erroneous usage. Note that our checking might not be comprehensive.

        info: information about all existing models (built-in and user-defined ones).
        deri_vari_dict: specify which attribute of which circuit element's will be calculated derivative w.r.t.
        node2ind, ind2node: information about the mapping from node's string name and node's index, and vice versa.
        node_element: specify which circuit elements are conneted to a node.
        inward_node, outward_node: information about the inward and outward direction of the port of a circuit element.

        """

        self.__node_signal = None

        # eval(repr().lower()) will convert every string to the corresponding lower case.
        try:
            self.circuit_element = eval(repr(circuit_element).lower())
            self.mode_info = eval(repr(mode_info).lower())
            self.omega = omega
            self.srce_node = eval(repr(srce_node).lower())
            self.prob_node = eval(repr(prob_node).lower())
            self.deri_node = eval(repr(deri_node).lower())
            self.deri_vari = eval(repr(deri_vari).lower())
            self.notation = eval(repr(notation).lower())
        except:
            raise RuntimeError("The provided inputs to initialize Circuit instance are problematic.")

        # The Circuit instance needs re-initialization after calling register_model(), so that self.info could be updated.
        with open(_model_json_name, 'r') as f:
            self.info = json.load(f)

        self._preprocess()

    def _preprocess(self):
        """Preprocess at the end of initialization

        We need to preprocess at the end of initialization to do some checking, and generate some useful variables. For
        details, please refer to each of the used functions. Moreover, this _preprocess function needs to be called after
        calling update_attr().

        """
        self.deri_vari_dict = self._deri_vari_processing()
        self.node2ind, self.ind2node, self.node_element, self.inward_node, self.outward_node = self._node_processing()

    def _check_name(self, index_string: str, attr_name: str, criterion: Optional[str] = 'identical') -> bool:
        """Check whether a given string occurs in an attribute of self.info

        This helper function is used to check the provided string by the user is correct. For instance, in the model.json
        file, we have a circuit element with model_name=wg, so that we have:

        _check_name('wgxxx', 'model_name', 'startswith') return True
        _check_name('wgxxx', 'model_name', 'identical') return False
        _check_name('wg', 'model_name', 'startswith') return True
        _check_name('wg', 'model_name', 'identical') return True

        :param index_string: a string that will be examined if it exists in self.info.
        :param attr_name: specify which attributes are of interests. It has to be those attributes in model.json, such as
                          'model_name', 'class_name', 'required_attr', 'optional_attr'.
        :param criterion: specify the comparison criterion.
        :return: True for occurrence, false for not.
        """
        if criterion not in ['identical', 'startswith']:
            raise RuntimeError("The comparison criterion is not implemented.")

        for entry in self.info:
            base_string = entry[attr_name]
            if criterion == 'startswith' and index_string.startswith(base_string):
                return True
            if criterion == 'identical' and base_string == index_string:
                return True

        return False

    def _deri_vari_processing(self) -> dict:
        """Preprocess the derivative information

        Process the attributes of Circuit class, and return a dictionary about which circuit element's attribute will be
         calculated derivative w.r.t. The dictionary key represents the circuit element, and the dictionary value
         represents the attribute. A few examinations are performed in this function to eliminate the possibility of
         erroneous usage of the simulator, but the readers should be warned that our checking is not comprehensive.

        :return: A dictionary with circuit element as key, and its attribute as value.
        """
        deri_dict = {}
        for cur_deri_vari in self.deri_vari:
            try:
                key, value = cur_deri_vari.split("::")
            except:
                raise RuntimeError("The provided derivative variable '%s' is problematic." % cur_deri_vari)

            if self._check_name(key, 'model_name', 'startswith'):
                if key not in self.circuit_element.keys():
                    raise RuntimeError(
                        "The element named '%s' (required by '%s') doesn't present in the circuit (i.e., circuit_element)." % (
                            key, cur_deri_vari))
                elif key not in deri_dict.keys():
                    # We postpone the checking of whether 'value' exists to when we calculate the derivative.
                    deri_dict[key] = [value]
                elif value in deri_dict[key]:
                    raise RuntimeError("The derivative variable '%s' is replicated." % cur_deri_vari)
                else:
                    # We postpone the checking of whether 'value' exists to when we calculate the derivative.
                    deri_dict[key].append(value)
            else:
                raise RuntimeError("The circuit model of '%s' (required by '%s') is not defined. Please check if it "
                                   "is a typo." % (key, cur_deri_vari))

        return deri_dict

    def _node_processing(self) -> Tuple[dict, dict, dict, dict, dict]:
        """Preprocess the node information

        Process the attributes of Circuit class, and return several dictionaries about the mappings of nodes. The nodes
        are specified in strings in the netlist, while to run the simulation, we need to assign each node an index, so
         that a matrix equation could be built. We need to store the forward mapping from node string name to node index
          (node2ind), as well as the backward mapping from node index to node string name (ind2node). Moreover, we also
          need to store information such as which circuit elements are connected to a node (node_element), and the inward
         and outward optical signals occupying which index (inward_node, outward_node). Except inward_node and outward_node,
         the other dictionaries should be easily understood. For these two dictionaries, please refer to the document on
         Principle of Simulator (PoS).

        :return: a tuple of dictionaries.
        """
        node_index, node2ind, ind2node, node_has_ele = 0, {}, {}, {}
        for ele, attr in self.circuit_element.items():
            nodes = attr['ln'] + attr['rn']
            for node in nodes:
                # update node2ind, ind2node
                if node not in node2ind:
                    node2ind[node] = node_index
                    ind2node[node_index] = node
                    node_index += 1
                # update node_has_ele
                if node not in node_has_ele:
                    node_has_ele[node] = [ele]
                else:
                    node_has_ele[node].append(ele)

        # build the inward node index dict and outward node index dict
        inward_node, outward_node = {}, {}
        for ele, attr in self.circuit_element.items():
            inward_node[ele], outward_node[ele] = {'ln': [], 'rn': []}, {'ln': [], 'rn': []}

            # ele_left_nodes, ele_right_nodes = attr['ln'], attr['rn']

            pair = zip(['ln', 'rn'], [attr['ln'], attr['rn']])

            for string, ele_nodes in pair:
                for node in ele_nodes:
                    if node_has_ele[node][0] == ele:
                        inward_node[ele][string].append(node2ind[node] * 2)
                        outward_node[ele][string].append(node2ind[node] * 2 + 1)
                    elif node_has_ele[node][1] == ele:
                        inward_node[ele][string].append(node2ind[node] * 2 + 1)
                        outward_node[ele][string].append(node2ind[node] * 2)
                    else:
                        raise RuntimeError("The node '%s' has more than two circuit elements connected." % (node))

            # for node in ele_left_nodes:
            #     if node_has_ele[node][0] == ele:
            #         inward_node[ele]['ln'].append(node2ind[node] * 2)
            #         outward_node[ele]['ln'].append(node2ind[node] * 2 + 1)
            #     elif node_has_ele[node][1] == ele:
            #         inward_node[ele]['ln'].append(node2ind[node] * 2 + 1)
            #         outward_node[ele]['ln'].append(node2ind[node] * 2)
            #     else:
            #         raise RuntimeError("The node '%s' has more than two circuit elements connected." % (node))
            # for node in ele_right_nodes:
            #     if node_has_ele[node][0] == ele:
            #         inward_node[ele]['rn'].append(node2ind[node] * 2)
            #         outward_node[ele]['rn'].append(node2ind[node] * 2 + 1)
            #     elif node_has_ele[node][1] == ele:
            #         inward_node[ele]['rn'].append(node2ind[node] * 2 + 1)
            #         outward_node[ele]['rn'].append(node2ind[node] * 2)
            #     else:
            #         raise RuntimeError("The node '%s' has more than two circuit elements connected." % (node))

        # As we have now processed all nodes, we could do a check if self.prob_nodes are correctly provided.

        if not set(self.srce_node.keys()).issubset(set(node2ind.keys())):
            raise RuntimeError("'srce_node' contains node not included in the circuit (i.e., circuit_element).")
        if not set(self.deri_node).issubset(set(node2ind.keys())):
            raise RuntimeError("'deri_node' contains node not included in the circuit (i.e., circuit_element).")
        if not set(self.prob_node).issubset(set(node2ind.keys())):
            raise RuntimeError("'prob_node' contains node not included in the circuit (i.e., circuit_element).")

        return node2ind, ind2node, node_has_ele, inward_node, outward_node

    def _build_matrix(self, A: np.ndarray, inward_node: List[int], outward_node: List[int], smatrix: np.ndarray,
                      line_counter: int, fillconst: float) -> Tuple[np.ndarray, int]:
        """Embed the relation (constraint) of one S matrix into the overall system matrix A.

        This is a helper function used by the solve method. A and smatrix is a three-dimensional np.ndarray. Both their
        shape[0] equals to the length of omega. Namely, A is a batched system matrix. Please refer to our document on
        Principle of Simulator (PoS) for details.

        :param A: the system matrix which the S matrix will be embedded into.
        :param inward_node: the node indices for the inward direction.
        :param outward_node: the node indices for the outward direction.
        :param smarix: the current S matrix.
        :param line_counter: specify which row the relation (constraint) will be at.
        :param fillconst: specify the filled value associated with the output node. It will be set to 1 when calculating
                          node magnitude response, and 0 when calculating derivative.
        :return: the updated system matrix A and the line counter.
        """
        for i in range(len(outward_node)):
            A[:, line_counter, outward_node[i]] = fillconst
            A[:, line_counter, inward_node] = smatrix[:, i, :]
            line_counter += 1
        return A, line_counter

    def solve(self, require_grads: Optional = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Solve the circuit port magnitude and derivative.

        This function is the core of the Circuit class. The solver logic is to build a large system matrix Ax=b at one
        frequency point, with A and b embedding all S-matrix relations of all circuit elements, and x is a (2N,1) unknown
        vector represents the complex port magnitudes. Here N represents the number of ports, because for each port, the
         optical signals have two propagating directions. Derivative is calculated based on a first order Taylor
        expansion: delta_A * x + A * delta_x = 0 => delta_x = -inv(A)* delta_A * x.

        In implementation, the port magnitudes and derivatives are calculated in a batch manner: all angular frequency
        points are solved at the same time. Namely, A is of size (len(omega), 2N, 2N), etc.

        :param require_grads: Whether gradient is calculated.
        :return: The port magnitude, and the gradient if needed.
        """
        require_grads = require_grads and self.deri_vari is not None and self.deri_node is not None \
                        and len(self.deri_vari) >= 1 and len(self.deri_node) >= 1

        line_counter, num_node = 0, len(self.ind2node.keys())

        module = importlib.import_module("spode.core.model")

        A, b = np.zeros([len(self.omega), 2 * num_node, 2 * num_node], dtype=np.complex_), \
               np.zeros([len(self.omega), 2 * num_node, 1], dtype=np.complex_)

        for node, ele in self.node_element.items():
            if len(ele) == 1:
                src_value = 0.0 if node not in self.srce_node.keys() else self.srce_node[node]

                # source must be injected at floating node, and the inward direction takes the first place
                A[:, line_counter, 2 * self.node2ind[node]] = 1.0
                b[:, line_counter, 0] = src_value
                line_counter += 1

        grads = [np.zeros([len(self.omega), 2 * num_node, 2 * num_node], dtype=np.complex_)
                 for _ in range(len(self.deri_vari))]

        for ele, attr in self.circuit_element.items():
            for entry in self.info:
                if ele.startswith(entry['model_name'].lower()):
                    class_ = getattr(module, entry['class_name'])
                    ele_instance = class_(**{**attr, **self.mode_info})
                    if ele in self.deri_vari_dict.keys():

                        S_l2r, grad_l2r_dict = ele_instance.get_smatrix(self.omega, direction='l2r',
                                                                        deri_vari=self.deri_vari_dict[ele])
                        S_r2l, grad_r2l_dict = ele_instance.get_smatrix(self.omega, direction='r2l',
                                                                        deri_vari=self.deri_vari_dict[ele])

                        for cur_deri_vari in self.deri_vari_dict[ele]:
                            if cur_deri_vari not in grad_r2l_dict.keys() or cur_deri_vari not in grad_l2r_dict.keys():
                                raise RuntimeError(
                                    "The gradient of S matrix w.r.t. '%s' in model '%s' is not implemented"
                                    % (cur_deri_vari, entry['class_name']))

                            line_counter_wrk = line_counter
                            index = self.deri_vari.index(ele + "::" + cur_deri_vari)
                            grads[index], line_counter_wrk = self._build_matrix(grads[index],
                                                                                self.inward_node[ele]['ln'],
                                                                                self.outward_node[ele]['rn'],
                                                                                grad_l2r_dict[cur_deri_vari],
                                                                                line_counter_wrk, 0)
                            grads[index], line_counter_wrk = self._build_matrix(grads[index],
                                                                                self.inward_node[ele]['rn'],
                                                                                self.outward_node[ele]['ln'],
                                                                                grad_r2l_dict[cur_deri_vari],
                                                                                line_counter_wrk, 0)

                    else:
                        S_l2r = ele_instance.get_smatrix(self.omega, direction='l2r', deri_vari=None)
                        S_r2l = ele_instance.get_smatrix(self.omega, direction='r2l', deri_vari=None)

                    A, line_counter = self._build_matrix(A, self.inward_node[ele]['ln'], self.outward_node[ele]['rn'],
                                                         S_l2r,
                                                         line_counter, -1.0)
                    A, line_counter = self._build_matrix(A, self.inward_node[ele]['rn'], self.outward_node[ele]['ln'],
                                                         S_r2l,
                                                         line_counter, -1.0)
                    break
            else:
                raise RuntimeError("The model %s is not defined in the simulator. Check if there is a typo, "
                                   "or define the model by yourself." % (ele))

        try:
            invA = np.linalg.inv(A)
        except:
            raise RuntimeError("Solving the system matrix fails. Please check the provided netlist.")

        response = np.matmul(invA, b)

        returned_res = np.zeros([len(self.prob_node), len(self.omega), 2], dtype=np.complex_)
        for i in range(len(self.prob_node)):
            # usually the probed node is a floating node, so the first place contains the complex magnitude for the
            # inward direction; the second place contains that for the outward direction.
            node_ind = self.node2ind[self.prob_node[i]]
            returned_res[i, :, 0] = response[:, 2 * node_ind, 0]
            returned_res[i, :, 1] = response[:, 2 * node_ind + 1, 0]

        self.__node_signal = response # (len(omega), #nodes, 2)

        if not require_grads:
            return returned_res

        returned_grads = np.zeros([len(self.deri_node), len(self.deri_vari), len(self.omega), 2], dtype=np.complex_)
        for j in range(len(self.deri_vari)):
            wrk_grad = np.matmul(np.matmul(-invA, grads[j]), response)
            for i in range(len(self.deri_node)):
                ind = self.node2ind[self.deri_node[i]]
                returned_grads[i, j, :, :] = wrk_grad[:, 2 * ind:2 * ind + 2, 0]

        return returned_res, returned_grads

    def update_attr(self, update_attr: str, new_value: Any, key_string: Optional[str] = None):
        """Update the attributes of the circuit

        Update the specified attribute with a provided new value. Since the Circuit class has both dict (even nested dict)
        and list attributes, this function provides a convenient interface to update any attribute. Example usage:

        circuit_instance.update_attr('mode_info', 4, "neff")
        circuit_instance.update_attr('deri_node', 'n222')
        circuit_instance.update_attr('circuit_element', ['n11', 'n222'], key_strings='tbum1::ln')
        circuit_instance.update_attr('circuit_element', 0.3 * np.pi, key_strings='ps1::ps')

        :param update_attr: the attribute that will be updated.
        :param new_value: the new value of the attribute.
        :param key_string: It specify the key, if the updated attribute is a dict.
        """
        update_attr = update_attr.lower()

        cur_value = getattr(self, update_attr)
        if isinstance(cur_value, dict):
            key_string_split = key_string.lower().split("::")
            reduce(operator.getitem, key_string_split[:-1], cur_value)[key_string_split[-1]] = new_value
            setattr(self, update_attr, cur_value)
        else:
            setattr(self, update_attr, new_value)

        # if the node or derivative variable is updated, we need to call _preprocess().
        self._preprocess()

    def get_attr(self, get_attr: str, key_strings: Optional[str] = None) -> Any:
        """Get the current value of some attribute.

        Get the current value of some attribute specified by get_attr (as well as key_strings if it exists). Example usage:

        circuit_instance.get_attr('circuit_element', 'tbum1::alpha')
        circuit_instance.get_attr('mode_info')

        :param get_attr: specify the value of which attribute we will get.
        :param: key_strings: If the desired attribute is the value in a dict, the key needs to be provided.
        :return: the value of the attribute.
        """
        get_attr = get_attr.lower()

        cur_value = getattr(self, get_attr)
        if isinstance(cur_value, dict):
            key_strings = key_strings.lower().split("::")
            return reduce(operator.getitem, key_strings, cur_value)
        return cur_value

    def visualize(self, placement: Optional[str] = '', cell_place_function: Callable = None, **kwargs):
        """Visualize the Circuit instance

        :param placement: a string specified the placement of the provided circuit. It will be used only if
                      cell_place_function is None. :param
        :param cell_place_function: A callable function, given the cell name, length and width of TBU, return the position of
                        cell center.
        :param kwargs: a set of parameters determining the visualization. Accepted parameters including:

        length_width_ratio: The ratio of TBU length over TBU width
        add_on_ratio: a tuple with two float numbers, the first for length of add-on in the 'TBU width' direction, the
                         second for length of add-on in the 'TBU length' direction.
        annotate: a string represents the annotation style.
        title: the title of the figure.
        line2d_property: a dictionary with properties accepted by plt.plot()
        polygon_property: a dictionary with properties accepted by plt.fill()

        """

        if placement == '' and cell_place_function is None:
            raise RuntimeError("One of 'notation' and 'cell_place_function' should be provided to do visualization.")

        cell_place_function = cell_place_function if cell_place_function is not None else _cell_place_function_list(
            placement)
        _visualize_base(self.circuit_element, cell_place_function,
                        node_signal=self.__node_signal,
                        node2ind=self.node2ind, **kwargs)


if __name__ == '__main__':
    circuit_element = {
        'tbum1': {'Ln': ['N0', 'n1'], 'rn': ['n2', 'N3'], 'theta': 0.0 * np.pi, 'pHi': 0.0 * np.pi, 'L': 250e-6},
        'wg1': {'ln': ['n3'], 'rn': ['n4'], 'L': 0, 'alpha': 1.0},
        'pS1': {'ln': ['n4'], 'rn': ['n5'], 'ps': 0.0 * np.pi}}

    mode_info = {'neFf': 2.35, 'wl': 1550e-9}
    omega = np.linspace(192.5, 193.5, 3) * 1e12 * 2 * np.pi
    prob_node = ['N5']
    deri_node = ['n5']
    srce_node = {'N0': 1.0}
    deri_vari = ['tbum1::theta', 'wG1::l']

    circ_instance = Circuit(circuit_element=circuit_element,
                            mode_info=mode_info,
                            omega=omega,
                            srce_node=srce_node,
                            prob_node=prob_node,
                            deri_node=deri_node,
                            deri_vari=deri_vari)

    # print(circ_instance.node2ind)
    # print(circ_instance.ind2node)
    # print(circ_instance.node_element)
    # print(circ_instance.inward_node)
    # print(circ_instance.outward_node)

    returned = circ_instance.solve()

    print("hello")

    # circ_instance.update_attr('mode_info', 4, "neff")
    # circ_instance.update_attr('deri_node', 'n222')
    # circ_instance.update_attr('circuit_element', ['n11', 'n222'], key_strings='tbum1::ln')
    # circ_instance.update_attr('circuit_element', 0.3 * np.pi, key_strings='ps1::ps')
    #
    # circ_instance.get_attr('mode_info', "neff")
    # circ_instance.get_attr('mode_info', "neff")
    # circ_instance.get_attr('circuit_element', key_strings='tbum1::ln')
    # circ_instance.get_attr('circuit_element', key_strings='ps1::ps')
