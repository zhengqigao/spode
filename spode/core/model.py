import numpy as np
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
from abc import abstractmethod, ABCMeta
import json
import warnings
from collections import OrderedDict
import importlib
import os

import spode.core.const as const


# the model json file. It will be used as reference when parsing the circuit from a netlist file
_model_json_name = os.path.join(os.path.dirname(__file__), 'model.json')

# the built-in models defined in this file.
_all_class_names = ['WaveGuide', 'PhaseShift', 'DirectionalCoupler', 'TBUm', 'TBUo', 'TBUt']

# the models and methods which are visible to users
__all__ = ['GeneralModel'] + _all_class_names + ['reset_model_json', 'register_model']


def _neff(neff0: float, ng0: Optional[float] = None, wl0: Optional[float] = None) -> Callable:
    """Approximate neff(omega) via first-order Taylor expansion based on {neff,ng} provided at a specific wavelength

    There are two ways to call this function: (i) If dispersion effect is not considered, we use _neff(neff0), or (ii)
    if dispersion effect is considered, we call _neff(neff0, ng0, wl0). Note that in case (ii), we must provide group
    index ng0, as well as the wavelength wl0 where {neff0,ng0} are provided.

    :param neff: The effective index at the specified wavelength.
    :param ng: The group index at the specified wavelength.
    :param wl: The wavelength where neff and ng are provided.
    :return: An anonymous function, taking an angular frequency point as input, returning the corresponding neff.
    """

    if wl0 == None:  # If ng is not provided, neff is a constant independent of frequency
        neff_func = lambda omega: neff0 * np.ones_like(omega)
    else:  # If ng is provided, neff is frequency dependent
        def neff_func(omega):
            wl_e = 2 * np.pi * const.FreeLightSpeed / omega
            return wl_e / wl0 * neff0 + (1 - wl_e / wl0) * ng0
    return neff_func


class GeneralModel(object, metaclass=ABCMeta):
    """A generic model (abstract base class)

    Users could define their own circuit element by inheriting this abstract class, and override the `__init__'
    method and the `get_smatrix' method. Thus, this differentiable simulator could be expanded to incorporate various
    customized circuit elements, and ultimately suits not only to programmable photonics, but also more generally
    integrated photonics.

    Class Variables:
        _name: the model name. It will be used to parse the model defined in the circuit netlist.
        _required_attr: mandatory attributes of the circuit model. Their values must be provided.
        _optional_attr: optional attributes of the circuit model. If omitted, they will be set to default values.
        _differential_attr: the attributes that the derivatives could be evaluated w.r.t.
        _num_port: a list of two elements, the number of ports at the left end and right end.

    Instance Variables:
        params: the circuit model parameters and the associated values in the format of key-value pair.
    """

    # the following parameters are class variables. They should be determined according to the given circuit model.
    # They shouldn't be accessed and changed in the program once written.

    _name = ''
    _required_attr = []
    _optional_attr = {}
    _differential_attr = []
    _num_port = [None, None]

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize GeneralModel

        Initialization method. Need to be overridden in the child class.

        Attributes:
            params: the circuit model parameters and the associated values in the format of key-value pair.
        """

        self.params = {}

        for attr in self._required_attr:
            if attr in kwargs.keys():
                self.params[attr] = kwargs[attr]
            else:
                raise RuntimeError("The value of attribute '%s' is not provided when initializing model '%s'."
                                   % (attr, self.__class__.__name__))

        for attr, default_value in self._optional_attr.items():
            self.params[attr] = kwargs[attr] if attr in kwargs.keys() else default_value

    @abstractmethod
    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:
        """Calculate the scattering matrix and gradient matrix at specified angular frequency points.

        :param omega: Angular frequency points that the scattering and the gradient matrices are evaluated at.
        :param direction: The direction used to calculate the matrices.
        :param deri_vari: The variables that gradients are calculated with respect to. Default value is None, no gradient
                          is calculated. 'all' is accepted as a convenient way to specify gradients are returned w.r.t.
                          all variables; Otherwise, it should be a list of strings.
        :return: The calculated scattering matrix (as well as the gradient matrix, if deri_vari is not None.). It should
                 be of size (len(omega), self._num_port[0], self._num_port[1]) if direction == 'l2r'; otherwise, if
                 direction == 'r2l', it should be (len(omega), self._num_port[1], self._num_port[0]).
        """

    def set_attr(self, attr: str, value: Any) -> None:
        """Set the specified attribute of the circuit model to a provided value.

        :param attr: The attribute of the circuit model which will be updated.
        :param value: The new value that the attribute will be set to.
        :return: None.
        """

        self.params[attr] = value

    def get_attr(self, attr: str) -> Any:  # Float might be a more accurate description of current returned type hint.
        """Get the attribute of the circuit model.

        :param attr: The concerned attribute.
        :return: The value of the concerned attribute.
        """
        return self.params[attr]

    def print_allattr(self) -> None:
        """Print the values of all attributes defined in the circuit model.

        :return: None
        """
        for attr, value in self.params.items():
            if value is not None:
                if isinstance(value, str):
                    print("Attribute %s = %s" % (attr, value))
                else:
                    print("Attribute %s = %.2e" % (attr, value))
            else:
                print("Attribute %s = None" % attr)

    @classmethod
    def _collect_info(cls) -> dict:
        """ Collect model information

        This function is a convenient way to collect the information stored in the attributes start with an underscore.
        When a new circuit model is generated, the circuit model description file 'models.json' need to be updated, so
        that when the user define the new model in the netlist, the parser could correctly recognize it. Namely, if the
        user have written a new circuit model, this method will be called. Otherwise, it won't be used.

        :return: A dictionary containing all information, which could be later converted into a json file.
        """

        info_dict = OrderedDict()
        info_dict['model_name'] = cls._name
        info_dict['class_name'] = cls.__name__
        info_dict['required_attr'] = [attr for attr in cls._required_attr]
        info_dict['optional_attr'] = [attr for attr in cls._optional_attr]
        info_dict['differential_attr'] = cls._differential_attr
        info_dict['num_port'] = cls._num_port

        return info_dict


class WaveGuide(GeneralModel):
    """The class definition of a waveguide model."""

    ### Model parameters meaning ###
    # L (m): waveguide length
    # neff: effective index of the propagating model
    # ng: group index
    # wl (m): the free space wavelength where neff and ng are provided.
    # alpha: loss term
    ################################

    _name = 'wg'
    _required_attr = ['L', 'neff']
    _optional_attr = {'ng': 0, 'wl': None, 'alpha': 1.0}
    _differential_attr = ['L', 'alpha']
    _num_port = [1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Inheriting the initialization method, but need to deal with ng.
        if 'ng' in kwargs.keys():
            self.params['ng'] = kwargs['ng']
        else:
            self.params['ng'] = self.params['neff']

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        # linearly approximate the effective index neff
        neff_func = _neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = (omega * neff_func(omega) / const.FreeLightSpeed).reshape(-1,1,1)  # propagation constant

        # define and reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])
        # This model doesn't use the variable direction.
        Smatrix = self.params['alpha'] * np.exp(1.j * beta * self.params['L']).reshape(-1, 1, 1)

        if deri_vari is not None:
            grad = {}

            if 'L' in deri_vari:
                grad['L'] = 1.j * beta * Smatrix

            if 'alpha' in deri_vari:
                grad['alpha'] = 1.0 / self.params['alpha'] * Smatrix

            return Smatrix, grad
        else:
            return Smatrix


class PhaseShift(GeneralModel):
    """The class definition of a pure phase shift model."""

    ### Model parameters meaning ###
    # ps (rad): phase shift imposed when adding the phase shift element.
    ################################

    _name = 'ps'
    _required_attr = ['ps']
    _optional_attr = {}
    _differential_attr = ['ps']
    _num_port = [1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        # define and reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])
        # This model doesn't use the variable direction.
        Smatrix = np.exp(1.j * self.params['ps']) * np.ones([len(omega), 1, 1])

        if deri_vari is not None:
            grad = {}

            if 'ps' in deri_vari:
                grad['ps'] = 1.j * Smatrix

            return Smatrix, grad
        else:
            return Smatrix


class DirectionalCoupler(GeneralModel):
    """The class definition of a directional coupler model."""

    ### Model parameters meaning ###
    # cp_angle (rad): the angle of coupling, i.e., sin^2(cp_angle) is power coupling ratio.
    # alpha: loss term.
    ################################

    _name = 'dc'
    _required_attr = ['cp_angle']
    _optional_attr = {'alpha': 1.0}
    _differential_attr = ['cp_angle', 'alpha']
    _num_port = [2, 2]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        theta = self.params['cp_angle']
        Smatrix = self.params['alpha'] * np.array([[np.cos(theta), 1.j * np.sin(theta)],
                                                   [1.j * np.sin(theta), np.cos(theta)]], dtype=np.complex_)

        # reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])
        # This model doesn't use the variable direction.
        Smatrix = np.expand_dims(Smatrix, axis=0).repeat(len(omega), axis=0)

        if deri_vari is not None:
            grad = {}

            if 'cp_angle' in deri_vari:
                grad_cp_angle = self.params['alpha'] * np.array([[-np.sin(theta), 1.j * np.cos(theta)],
                                                                 [1.j * np.cos(theta), -np.sin(theta)]],
                                                                dtype=np.complex_)
                grad['cp_angle'] = np.expand_dims(grad_cp_angle, axis=0).repeat(len(omega), axis=0)

            if 'alpha' in deri_vari:
                grad['alpha'] = Smatrix / self.params['alpha']

            return Smatrix, grad
        else:
            return Smatrix


class TBUm(GeneralModel):
    """The class definition of a tunable basic unit (tbu) model with all phase shifts in the middle."""
    #TODO: add paper refernce, who uses this model.

    _name = 'tbum'
    _required_attr = ['theta', 'phi', 'L', 'neff']
    _optional_attr = {'ng': 0, 'wl': None, 'alpha': 1.0, 'cp_left': 0.25 * np.pi, 'cp_right': 0.25 * np.pi}
    _differential_attr = ['theta', 'phi', 'L', 'cp_left', 'cp_right']
    _num_port = [2, 2]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        # linearly approximate the effective index neff
        neff_func = _neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = omega * neff_func(omega) / const.FreeLightSpeed  # propagation constant

        # define and reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])

        S1 = np.zeros((2, 2), dtype=np.complex_)
        S1[0, 0] = S1[1, 1] = 1.0 * np.cos(self.params['cp_left'])
        S1[0, 1] = S1[1, 0] = 1.j * np.sin(self.params['cp_left'])

        S2 = np.zeros((2, 2), dtype=np.complex_)
        S2[0, 0] = np.exp(1.j * self.params['theta'])
        S2[1, 1] = np.exp(1.j * self.params['phi'])

        S3 = np.zeros((2, 2), dtype=np.complex_)
        S3[0, 0] = S3[1, 1] = 1.0 * np.cos(self.params['cp_right'])
        S3[0, 1] = S3[1, 0] = 1.j * np.sin(self.params['cp_right'])

        # If the left and right directional couplers are not identical, then the S matrix will be different.
        if direction == 'r2l':
            S = S1 @ S2 @ S3
            gradS_theta = S1 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S3
            gradS_phi = S1 @ np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]]) @ S3
        elif direction == 'l2r':
            S = S3 @ S2 @ S1
            gradS_theta = S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = S3 @ np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]]) @ S1
        else:
            S = S3 @ S2 @ S1
            gradS_theta = S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = S3 @ np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]]) @ S1
            # TODO: the warning message printed is bit messy; could be improved.
            warnings.warn("The variable 'direction' should be either 'l2r' or 'r2l' when building S matrix. "
                          "Automatically use 'direction'='l2r'. This might lead to error, if the "
                          "two directional couplers in the {} are not identical.".format(self._name))

        S_wg = (np.exp(1.j * beta * self.params['L']) * self.params['alpha']).reshape(-1, 1, 1)

        Smatrix = S_wg * np.expand_dims(S, axis=0).repeat(len(omega), axis=0)

        if deri_vari is not None:
            grad = {}

            if 'theta' in deri_vari:
                grad['theta'] = S_wg * gradS_theta

            if 'phi' in deri_vari:
                grad['phi'] = S_wg * gradS_phi

            if 'alpha' in deri_vari:
                grad['alpha'] = Smatrix / self.params['alpha']

            if 'L' in deri_vari:
                grad['L'] = Smatrix * 1.j * beta

            return Smatrix, grad
        else:
            return Smatrix

class TBUo(GeneralModel):
    """The class definition of a tunable basic unit (tbu) model with phase shifts in the middle and right,
    but in one row.

    This model has been used by some papers, such as S. Bandyopadhyay et al., 'Hardware error correction for
    programmable photonics', Optica.

    """

    _name = 'tbuo'
    _required_attr = ['theta', 'phi', 'L', 'neff']
    _optional_attr = {'ng': 0, 'wl': None, 'alpha': 1.0, 'cp_left': 0.25 * np.pi, 'cp_right': 0.25 * np.pi}
    _differential_attr = ['theta', 'phi', 'L', 'cp_left', 'cp_right']
    _num_port = [2, 2]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        # linearly approximate the effective index neff
        neff_func = _neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = omega * neff_func(omega) / const.FreeLightSpeed  # propagation constant

        # define and reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])

        S1 = np.zeros((2, 2), dtype=np.complex_)
        S1[0, 0] = S1[1, 1] = 1.0 * np.cos(self.params['cp_left'])
        S1[0, 1] = S1[1, 0] = 1.j * np.sin(self.params['cp_left'])

        S2 = np.zeros((2, 2), dtype=np.complex_)
        S2[0, 0] = np.exp(1.j * self.params['theta'])
        S2[1, 1] = 1.0

        S3 = np.zeros((2, 2), dtype=np.complex_)
        S3[0, 0] = S3[1, 1] = 1.0 * np.cos(self.params['cp_right'])
        S3[0, 1] = S3[1, 0] = 1.j * np.sin(self.params['cp_right'])

        S4 = np.zeros((2, 2), dtype=np.complex_)
        S4[0, 0] = np.exp(1.j * self.params['phi'])
        S4[1, 1] = 1.0

        # If the left and right directional couplers are not identical, then the S matrix will be different.
        if direction == 'r2l':
            S = S1 @ S2 @ S3 @ S4
            gradS_theta = S1 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S3 @ S4
            gradS_phi = S1 @ S2 @ S3 @ np.array([[1.j * np.exp(1.j * self.params['phi']), 0.], [0., 0.]])
        elif direction == 'l2r':
            S = S4 @ S3 @ S2 @ S1
            gradS_theta = S4 @ S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = np.array([[1.j * np.exp(1.j * self.params['phi']), 0.], [0., 0.]]) @ S3 @ S2 @ S1
        else:
            S = S4 @ S3 @ S2 @ S1
            gradS_theta = S4 @ S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = np.array([[1.j * np.exp(1.j * self.params['phi']), 0.], [0., 0.]]) @ S3 @ S2 @ S1
            warnings.warn("The variable 'direction' should be either 'l2r' or 'r2l' when building S matrix. "
                          "Automatically use 'direction'='l2r'. This might lead to error, if the "
                          "two directional couplers in the {} are not identical.".format(self._name))

        S_wg = (np.exp(1.j * beta * self.params['L']) * self.params['alpha']).reshape(-1, 1, 1)

        Smatrix = S_wg * np.expand_dims(S, axis=0).repeat(len(omega), axis=0)

        if deri_vari is not None:
            grad = {}

            if 'theta' in deri_vari:
                grad['theta'] = S_wg * gradS_theta

            if 'phi' in deri_vari:
                grad['phi'] = S_wg * gradS_phi

            if 'alpha' in deri_vari:
                grad['alpha'] = Smatrix / self.params['alpha']

            if 'L' in deri_vari:
                grad['L'] = Smatrix * 1.j * beta

            return Smatrix, grad
        else:
            return Smatrix

class TBUt(GeneralModel):
    """The class definition of a tunable basic unit (tbu) model with phase shifts in the middle and right,
    but in two rows."""

    # TODO: add paper reference, who uses this model.

    _name = 'tbut'
    _required_attr = ['theta', 'phi', 'L', 'neff']
    _optional_attr = {'ng': 0, 'wl': None, 'alpha': 1.0, 'cp_left': 0.25 * np.pi, 'cp_right': 0.25 * np.pi}
    _differential_attr = ['theta', 'phi', 'L', 'cp_left', 'cp_right']
    _num_port = [2, 2]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_smatrix(self,
                    omega: Union[list, np.ndarray],
                    direction: Optional[str] = None,
                    deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
            Union[Tuple[np.ndarray, dict], np.ndarray]:

        # linearly approximate the effective index neff
        neff_func = _neff(self.params['neff'], self.params['ng'], self.params['wl'])
        beta = omega * neff_func(omega) / const.FreeLightSpeed  # propagation constant

        # define and reshape the Smatrix to the correct shape (len(omega), self._num_port[0], self._num_port[1])

        S1 = np.zeros((2, 2), dtype=np.complex_)
        S1[0, 0] = S1[1, 1] = 1.0 * np.cos(self.params['cp_left'])
        S1[0, 1] = S1[1, 0] = 1.j * np.sin(self.params['cp_left'])

        S2 = np.zeros((2, 2), dtype=np.complex_)
        S2[0, 0] = np.exp(1.j * self.params['theta'])
        S2[1, 1] = 1.0

        S3 = np.zeros((2, 2), dtype=np.complex_)
        S3[0, 0] = S3[1, 1] = 1.0 * np.cos(self.params['cp_right'])
        S3[0, 1] = S3[1, 0] = 1.j * np.sin(self.params['cp_right'])

        S4 = np.zeros((2, 2), dtype=np.complex_)
        S4[0, 0] = 1.0
        S4[1, 1] = np.exp(1.j * self.params['phi'])

        # If the left and right directional couplers are not identical, then the S matrix will be different.
        if direction == 'r2l':
            S = S1 @ S2 @ S3 @ S4
            gradS_theta = S1 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S3 @ S4
            gradS_phi = S1 @ S2 @ S3 @ np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]])
        elif direction == 'l2r':
            S = S4 @ S3 @ S2 @ S1
            gradS_theta = S4 @ S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]]) @ S3 @ S2 @ S1
        else:
            S = S4 @ S3 @ S2 @ S1
            gradS_theta = S4 @ S3 @ np.array([[1.j * np.exp(1.j * self.params['theta']), 0.], [0., 0.]]) @ S1
            gradS_phi = np.array([[0., 0.], [0., 1.j * np.exp(1.j * self.params['phi'])]]) @ S3 @ S2 @ S1
            warnings.warn("The variable 'direction' should be either 'l2r' or 'r2l' when building S matrix. "
                          "Automatically use 'direction'='l2r'. This might lead to error, if the "
                          "two directional couplers in the {} are not identical.".format(self._name))

        S_wg = (np.exp(1.j * beta * self.params['L']) * self.params['alpha']).reshape(-1, 1, 1)

        Smatrix = S_wg * np.expand_dims(S, axis=0).repeat(len(omega), axis=0)

        if deri_vari is not None:
            grad = {}

            if 'theta' in deri_vari:
                grad['theta'] = S_wg * gradS_theta

            if 'phi' in deri_vari:
                grad['phi'] = S_wg * gradS_phi

            if 'alpha' in deri_vari:
                grad['alpha'] = Smatrix / self.params['alpha']

            if 'L' in deri_vari:
                grad['L'] = Smatrix * 1.j * beta

            return Smatrix, grad
        else:
            return Smatrix

def reset_model_json() -> bool:
    """Reset the model json file to only reflect the built-in models defined in this file."""
    info_list = []
    module = importlib.import_module("spode.core.model")
    for model_name in _all_class_names:
        class_ = getattr(module, model_name)
        info_list.append(class_._collect_info())

    with open(_model_json_name, 'w') as f:
        json.dump(info_list, f, indent=4, separators=(',', ':'))

    print("model.json file has been reset to default successfully.")

    return True


def register_model(model: Type[GeneralModel]) -> bool:
    """Register a user-defined model to the model.json file.

    Since the provided netlist will be parsed according to the model.json file. This function will correctly register
    the user-defined circuit model, so that during parsing, it could be recognized.

    :param model: The name of the class (e.g., WaveGuide, TBUm).
    """
    # check if the class name or the _name is used by other already defined models
    module = importlib.import_module("spode.core.model")
    for defined_model in _all_class_names:
        defined_class = getattr(module, defined_model)
        if model.__name__.lower() == defined_class.__name__.lower():
            raise RuntimeError("The class name %s is already used" % defined_class.__name__)
        if model._name.lower() == defined_class._name.lower():
            raise RuntimeError("The model name %s is already used" % defined_class._name)

    with open(_model_json_name, 'r') as f:
        info = json.load(f)

    info.append(model._collect_info())

    with open(_model_json_name, 'w') as f:
        json.dump(info, f, indent=4, separators=(',', ':'))

    print("The new model '%s' has been added successfully." % (model.__name__))
    return True


if __name__ == '__main__':
    print("hello world. This is the wrong file to run. "
          "If you insist, it will do a testing run.")

    omega = np.linspace(0, 10, 10)
    wg = WaveGuide(L=3., neff=2.)
    wg.print_allattr()
    S1 = wg.get_smatrix(omega)
    print(S1.shape)
    print('---')

    ps = PhaseShift(ps=0.1 * np.pi)
    ps.print_allattr()
    S2 = ps.get_smatrix(omega)
    print(S2.shape)
    print('---')

    dc = DirectionalCoupler(**{'cp_angle': 0.2 * np.pi})
    dc.print_allattr()
    S3 = dc.get_smatrix(omega)
    print(S3.shape)
    print('---')

    tbum = TBUm(**{'theta': 0., 'phi': 0., 'L': 250e-6, 'neff': 2.35})
    tbum.print_allattr()
    S4 = tbum.get_smatrix(omega)
    print(S4.shape)
    print('---')

    reset_model_json()


    class test(GeneralModel):
        _name = 'testuser'
        _required_attr = ['a1', 'a2']
        _optional_attr = ['a3', 'a4']
        _differential_attr = ['a4']
        _num_port = [2, 2]

        def __init__(self):
            super().__init__()

        def get_smatrix(self,
                        omega: Union[list, np.ndarray],
                        direction: Optional[str] = None,
                        deri_vari: Optional[Union[str, List[str], Set[str]]] = None) -> \
                Union[Tuple[np.ndarray, dict], np.ndarray]:
            return np.array([1])


    register_model(test)

    reset_model_json()
