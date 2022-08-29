import numpy as np
import matplotlib.pyplot as plt
from spode.core.solver import Circuit
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
from spode.util.generator import _config_size_dict, Generator
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors

__all__ = ['visualize']

def _cell_place_function_list(placement: str):
    if placement.lower() == 'square_1':
        def cell_place_function(x, tbu_length, tbu_width):
            return np.array([tbu_width * (x[0] + 1) + tbu_length * (x[0] + 0.5),
                             tbu_width * (x[1] + 1) + tbu_length * (x[1] + 0.5)])

    return cell_place_function


def visualize(circuit: Union[dict, Type[Circuit]],
              notation: Optional[str] = '',
              cell_place_function: Callable = None,
              **kwargs):
    if isinstance(circuit, dict):
        if cell_place_function is not None:
            _visualize_base(circuit, cell_place_function, **kwargs)
        else:
            cell_place_function = _cell_place_function_list(notation)
            _visualize_base(circuit, cell_place_function, **kwargs)
    elif isinstance(circuit, Circuit):
        print("hello world")
    else:
        raise RuntimeError(
            "The object being visualized must be a dict containing circuit elements or a Circuit instance.")


def _abc_through_two_points(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Calculate {a,b,c} of ax+by+c=0 going through two given points.

    :param point1: np.array (2,1) or (2,). Coordinate of point1.
    :param point2: np.array (2,1) or (2,). Coordinate of point2.
    :return: np.array (3,). {a,b,c} in order.
    """

    x1, y1 = point1.flatten()
    x2, y2 = point2.flatten()

    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    ratio = (a ** 2 + b ** 2 + c ** 2) ** 0.5
    a, b, c = a / ratio, b / ratio, c / ratio

    return np.array([a, b, c])


def _perpendicular_one_line_through_one_point(line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Calculate {a,b,c} of ax+by+c = 0 perpendicular a given line and going through a given point

    :param line: np.array (3,1) or (3,). {a,b,c} of the given line.
    :param point: np.array (2,1) or (2,). Coordinate of the given point.
    :return: np.array (3,). {a,b,c} in order
    """
    line = line.flatten()
    a, b = -line[1], line[0]
    c = - (a * point[0] + b * point[1])

    ratio = (a ** 2 + b ** 2 + c ** 2) ** 0.5
    a, b, c = a / ratio, b / ratio, c / ratio

    return np.array([a, b, c])


def _get_two_edges(tbu_length, tbu_width, cell1_center, cell2_center):
    middle_point = (cell1_center + cell2_center) / 2.0
    a, b, c = _abc_through_two_points(cell1_center, cell2_center)

    # go along the line connecting cell1_center and cell2_center
    t = tbu_width / 2.0 / (a ** 2 + b ** 2) ** 0.5
    point1 = middle_point - t * np.array([b, -a])
    point2 = middle_point + t * np.array([b, -a])

    # go along the direction perpendicular to the line connecting cell1_center and cell2_center
    t = tbu_length / 2 / (a ** 2 + b ** 2) ** 0.5
    edge1 = np.array([point1 + t * np.array([a, b]), point1 - t * np.array([a, b])])
    edge2 = np.array([point2 + t * np.array([a, b]), point2 - t * np.array([a, b])])

    return edge1, edge2


def _visualize_base(circuit_element: dict,
                    cell_place_func: Callable,
                    length_width_ratio: Optional[float] = 0.12,
                    add_on_ratio: Optional[dict] = (0.4, 0.6),
                    title: Optional[str] = '',
                    line2d_property: Optional[dict] = None,
                    polygon_property: Optional[dict] = None):
    """"A base function for visualization

    :param circuit_element: A dictionary contains all TBUs and its connections
    :param length_width_ratio: The ratio of TBU length over TBU width
    :param add_on_ratio: a tuple with two float numbers, the first for length of add-on in the 'width' direction, the
                         second for length of add-on in the 'length' direction.
    :param title: the title of the figure.
    :param line2d_property: a dictionary contains all properties accepted by plt.plot()
    :param polygon_property: a dictionary contains all properties accepted by plt.fill()

    :return: None.

    """
    tbu_lenth = 1
    tbu_width = tbu_lenth * length_width_ratio
    add_on_ratio_width, add_on_ratio_length = add_on_ratio

    ax = plt.figure(figsize=[8, 8])
    ax.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')

    for tbu, value in circuit_element.items():
        # as an example, tbu = tbum1_1#2_1#3_v
        tbu_name, cell1_name, cell2_name, *_ = tbu.split("_")
        cell1_center = cell_place_func([int(ele) for ele in cell1_name.split('#')])
        cell2_center = cell_place_func([int(ele) for ele in cell2_name.split('#')])

        # plot the background two edges, default color is mediumslateblue.
        edge1, edge2 = _get_two_edges(tbu_lenth, tbu_width, cell1_center, cell2_center)
        plt.plot(edge1[:, 0], edge1[:, 1], mcolors.CSS4_COLORS['mediumslateblue'], **line2d_property)
        plt.plot(edge2[:, 0], edge2[:, 1], mcolors.CSS4_COLORS['mediumslateblue'], **line2d_property)

        # plot the add-on rectangle, representing the phase shifts, default color is red.
        add_on_edge1, add_on_edge2 = _get_two_edges(tbu_lenth * (1 - add_on_ratio_length),
                                                    tbu_width * (1 + add_on_ratio_width), cell1_center, cell2_center)
        pts = np.concatenate((add_on_edge1, add_on_edge2), axis=0)
        hull = ConvexHull(pts)
        plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], 'red', **{**{'zorder': 100}, **polygon_property})

    if title:
        plt.title(title)

    plt.show()


# def _visualize_square_1(circuit_element: dict, cell_place_func: Callable, length_width_ratio: Optional[float] = 0.12,
#                         add_on_ratio: Optional[dict] = (0.4, 0.6), title: Optional[str] = '',
#                         line2d_property: Optional[dict] = None, polygon_property: Optional[dict] = None):
#     _visualize_base()


if __name__ == '__main__':
    # circuit_element = {
    #     'tbum1': {'Ln': ['N0', 'n1'], 'rn': ['n2', 'N3'], 'theta': 0.0 * np.pi, 'pHi': 0.0 * np.pi, 'L': 250e-6},
    #     'wg1': {'ln': ['n3'], 'rn': ['n4'], 'L': 0, 'alpha': 1.0},
    #     'pS1': {'ln': ['n4'], 'rn': ['n5'], 'ps': 0.0 * np.pi}}
    #
    # mode_info = {'neFf': 2.35, 'wl': 1550e-9}
    # omega = np.linspace(192.5, 193.5, 3) * 1e12 * 2 * np.pi
    # prob_node = ['N5']
    # deri_node = ['n5']
    # srce_node = {'N0': 1.0}
    # deri_vari = ['tbum1::theta', 'wG1::l']
    #
    # circ_instance = Circuit(circuit_element=circuit_element,
    #                         mode_info=mode_info,
    #                         omega=omega,
    #                         srce_node=srce_node,
    #                         prob_node=prob_node,
    #                         deri_node=deri_node,
    #                         deri_vari=deri_vari)
    # visualize(circ_instance)

    generator = Generator()
    circuit_element = generator.generate('square_1', [5, 5], init_dict={})
    _visualize_square_1(circuit_element, line2d_property={'linewidth': 3}, polygon_property={})

