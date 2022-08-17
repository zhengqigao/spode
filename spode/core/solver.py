import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
import importlib
import json

from spode.core.model import _model_json_name

__all__ = ['Circuit']


class Circuit(object):
    def __init__(self, circuit_element: dict, mode_info: dict, omega: Union[List[float], np.ndarray],
                 srce_node: dict, prob_node: Union[List[int], np.ndarray],
                 deri_node: Optional = None, deri_vari: Optional = None):

        self.circuit_element = circuit_element
        self.mode_info = mode_info
        self.omega = omega
        self.srce_node = srce_node
        self.prob_node = prob_node
        self.deri_node = deri_node
        self.deri_vari = deri_vari

        self.deri_vari_dict = self._deri_vari_processing()
        self.node2ind, self.ind2node, self.node_element, self.inward_node, self.outward_node = self._node_processing()

    def _deri_vari_processing(self):
        deri_dict = {}
        for deri_vari in self.deri_vari:
            key, value = deri_vari.split("::")
            if key not in deri_dict.keys():
                deri_dict[key] = [value]
            else:
                deri_dict[key].append(value)
        return deri_dict

    def _node_processing(self):
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
            ele_left_nodes, ele_right_nodes = attr['ln'], attr['rn']
            for node in ele_left_nodes:
                if node_has_ele[node][0] == ele:
                    inward_node[ele]['ln'].append(node2ind[node] * 2)
                    outward_node[ele]['ln'].append(node2ind[node] * 2 + 1)
                elif node_has_ele[node][1] == ele:
                    inward_node[ele]['ln'].append(node2ind[node] * 2 + 1)
                    outward_node[ele]['ln'].append(node2ind[node] * 2)
                else:
                    raise RuntimeError("The node '%s' has more than two circuit elements connected." % (node))
            for node in ele_right_nodes:
                if node_has_ele[node][0] == ele:
                    inward_node[ele]['rn'].append(node2ind[node] * 2)
                    outward_node[ele]['rn'].append(node2ind[node] * 2 + 1)
                elif node_has_ele[node][1] == ele:
                    inward_node[ele]['rn'].append(node2ind[node] * 2 + 1)
                    outward_node[ele]['rn'].append(node2ind[node] * 2)
                else:
                    raise RuntimeError("The node '%s' has more than two circuit elements connected." % (node))
        return node2ind, ind2node, node_has_ele, inward_node, outward_node

    def _build_matrix(self, A, inward_node, outward_node, smatrix, line_counter, fillconst):
        for i in range(len(outward_node)):
            A[:, line_counter, outward_node[i]] = fillconst
            A[:, line_counter, inward_node] = smatrix[:, i, :]
            line_counter += 1
        return A, line_counter

    def solve(self):
        require_grads = self.deri_vari is not None and self.deri_node is not None \
                        and len(self.deri_vari) >= 1 and len(self.deri_node) >= 1

        line_counter, num_node = 0, len(self.ind2node.keys())

        with open(_model_json_name, 'r') as f:
            info = json.load(f)
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

        if require_grads:
            grads = [np.zeros([len(self.omega), 2 * num_node, 2 * num_node], dtype=np.complex_)
                     for _ in range(len(self.deri_vari))]

        for ele, attr in self.circuit_element.items():
            for entry in info:
                if ele.startswith(entry['model_name'].lower()):
                    class_ = getattr(module, entry['class_name'])
                    ele_instance = class_(**{**attr, **self.mode_info})
                    if ele in self.deri_vari_dict.keys():

                        S_l2r, grad_l2r_dict = ele_instance.get_smatrix(self.omega, direction='l2r',
                                                                        deri_vari=self.deri_vari_dict[ele])
                        S_r2l, grad_l2r_dict = ele_instance.get_smatrix(self.omega, direction='r2l',
                                                                        deri_vari=self.deri_vari_dict[ele])

                        for cur_deri_vari in self.deri_vari_dict[ele]:
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
                                                                                grad_l2r_dict[cur_deri_vari],
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

        invA = np.linalg.inv(A)
        response = np.matmul(invA, b)

        returned_res = np.zeros([len(self.prob_node), len(self.omega), 2], dtype=np.complex_)
        for i in range(len(self.prob_node)):
            # usually the probed node is a floating node, so the first place contains the complex magnitude for the
            # inward direction; the second place contains that for the ourward direction.
            node_ind = self.node2ind[self.prob_node[i]]
            returned_res[i, :, 0] = response[:, 2 * node_ind, 0]
            returned_res[i, :, 1] = response[:, 2 * node_ind + 1, 0]

        if not require_grads:
            return returned_res

        returned_grads = np.zeros([len(self.deri_node), len(self.deri_vari), len(self.omega), 2], dtype=np.complex_)
        for j in range(len(self.deri_vari)):
            wrk_grad = np.matmul(np.matmul(-invA, grads[j]), response)
            for i in range(len(self.deri_node)):
                ind = self.node2ind[self.deri_node[i]]
                returned_grads[i, j, :, :] = wrk_grad[:, 2 * ind:2 * ind + 2, 0]

        return returned_res, returned_grads


if __name__ == '__main__':
    circuit_element = {
        'tbum1': {'ln': ['n0', 'n1'], 'rn': ['n2', 'n3'], 'theta': 0.0 * np.pi, 'phi': 0.0 * np.pi, 'L': 250e-6},
        'wg1': {'ln': ['n3'], 'rn': ['n4'], 'L': 0, 'alpha': 1.0},
        'ps1': {'ln': ['n4'], 'rn': ['n5'], 'ps': 0.0 * np.pi}}

    mode_info = {'neff': 2.35, 'ng': 4.0, 'wl': 1550e-9}
    omega = np.linspace(192.5, 193.5, 3) * 1e12 * 2 * np.pi
    prob_node = ['n5']
    deri_node = ['n5']
    srce_node = {'n0': 1.0}
    deri_vari = ['tbum1::theta', 'wg1::alpha', 'wg1::L']

    circ_instance = Circuit(circuit_element=circuit_element,
                            mode_info=mode_info,
                            omega=omega,
                            srce_node=srce_node,
                            prob_node=prob_node,
                            deri_node=deri_node,
                            deri_vari=deri_vari)

    print(circ_instance.node2ind)
    print(circ_instance.ind2node)
    print(circ_instance.node_element)
    print(circ_instance.inward_node)
    print(circ_instance.outward_node)

    returned = circ_instance.solve()

