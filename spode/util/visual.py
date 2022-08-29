import numpy as np
import matplotlib.pyplot as plt
from spode.core.solver import Circuit
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
from spode.util.generator import generate
from scipy.spatial import ConvexHull
import matplotlib.colors as mcolors

__all__ = ['visualize']


def _cell_place_function_list(placement: str) -> Callable:
    """Return a callable function calculating the coordinate of cell center based on the specified placement

    :param placement: a string represents the placement.
    :return: a callable function.
    """

    if 'square_1' == placement.lower():
        def cell_place_function(cell_name, tbu_length, tbu_width):
            x = ([int(ele) for ele in cell_name.split('#')])
            return np.array([tbu_width * (x[1] + 1) + tbu_length * (x[1] + 0.5),
                             tbu_width * (x[0] + 1) + tbu_length * (x[0] + 0.5)])
    elif 'hexagonal_1' == placement.lower():
        def cell_place_function(cell_name, tbu_length, tbu_width):
            layer_index, cell_index = ([int(ele) for ele in cell_name.split('#')])
            if layer_index == 1:
                return np.zeros(2, )
            unit_length = (3 ** 0.5) * tbu_length + tbu_width
            vertices = [1 + ele for ele in np.arange(7) * (layer_index - 1)]
            length = (layer_index - 1) * unit_length
            angles = np.array([np.pi, np.pi * 2 / 3.0, np.pi / 3.0, 0.0, -np.pi / 3.0, -np.pi * 2 / 3.0, -np.pi])
            coordinates = [length * np.array([np.cos(angle), -np.sin(angle)]) for angle in angles]
            for i in range(len(vertices) - 1):
                if vertices[i] <= cell_index < vertices[i + 1]:
                    start, end = coordinates[i], coordinates[i + 1]
                    ratio = (cell_index - vertices[i]) / (layer_index - 1)
                    x = start * (1 - ratio) + end * ratio
                    return x
    elif 'triangular_1' == placement.lower():
        def cell_place_function(cell_name, tbu_length, tbu_width):
            row_index, column_index = ([int(ele) for ele in cell_name.split('#')])
            unit_length = tbu_length / (3 ** 0.5) + tbu_width
            x = 0.5 * (3 ** 0.5) * (row_index - 1) * unit_length + (column_index - 1) * 0.5 * (3 ** 0.5) * unit_length
            y = 1.5 * (row_index - 1) * unit_length + 0.5 * (column_index % 2 == 0) * unit_length
            return np.array([x, y])
    else:
        raise RuntimeError("There is no built-in cell_place_function for this placement.")

    return cell_place_function


def visualize(circuit: Union[dict, Type[Circuit]],
              placement: Optional[str] = '',
              cell_place_function: Callable = None,
              **kwargs) -> None:
    """Visualize a programmable photonics circuit containing TBUs

    :param circuit: A dictionary contains all TBUs and its connections :param port_magnitude: A np.ndarray of shape (
                    #nodes, len(omega), 2), it will be used if the 'annotate' variable is specified.
    :param placement: a string specified the placement of the provided circuit. It will be used only if
                      cell_place_function is None. :param
    cell_place_function: A callable function, given the cell name, length and width of TBU, return the position of
                        cell center.
    :param kwargs: a set of parameters determining the visualization. Accepted parameters including:

        length_width_ratio: The ratio of TBU length over TBU width
        add_on_ratio: a tuple with two float numbers, the first for length of add-on in the 'TBU width' direction, the
                         second for length of add-on in the 'TBU length' direction.
        annotate: a string represents the annotation style.
        title: the title of the figure.
        line2d_property: a dictionary with properties accepted by plt.plot()
        polygon_property: a dictionary with properties accepted by plt.fill()

    :return: None.

    """

    if placement == '' and cell_place_function is None:
        raise RuntimeError("One of 'notation' and 'cell_place_function' should be provided to do visualization.")

    if isinstance(circuit, Circuit):
        circuit = circuit.circuit_element
    elif not isinstance(circuit, dict):
        raise RuntimeError(
            "The object being visualized must be a dict containing circuit elements or a Circuit instance.")

    cell_place_function = cell_place_function if cell_place_function is not None else _cell_place_function_list(
        placement)
    _visualize_base(circuit, cell_place_function, **kwargs)


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

    direction_vector = cell2_center - cell1_center
    # go along the line connecting cell1_center and cell2_center
    t = tbu_width / 2.0 / np.linalg.norm(direction_vector)
    point1 = middle_point - t * direction_vector
    point2 = middle_point + t * direction_vector

    # go along the direction perpendicular to the line connecting cell1_center and cell2_center
    a, b = -direction_vector[1], direction_vector[0]
    t = tbu_length / 2.0 / (a ** 2 + b ** 2) ** 0.5
    edge1 = np.array([point1 + t * np.array([a, b]), point1 - t * np.array([a, b])])
    edge2 = np.array([point2 + t * np.array([a, b]), point2 - t * np.array([a, b])])

    return edge1, edge2


def _visualize_base(circuit_element: dict, cell_place_func: Callable, **kwargs):
    """"A base function for visualization

    :param circuit_element: A dictionary contains all TBUs and its connections
    :param cell_place_func: A callable function, given the cell name, length and width of TBU, return the position of
                            cell center.
    :param kwargs: a set of parameters determining the visualization. Accepted parameters including:

        length_width_ratio: The ratio of TBU length over TBU width
        add_on_ratio: a tuple with two float numbers, the first for length of add-on in the 'width' direction, the
                    second for length of add-on in the 'length' direction.
        title: the title of the figure.
        line2d_property: a dictionary with properties accepted by plt.plot()
        polygon_property: a dictionary with properties accepted by plt.fill()
        text_property: a dictionary with properties accepted by plt.text()
    :return: None.

    """

    if 'length_width_ratio' not in kwargs.keys():
        length_width_ratio = 0.12
    else:
        length_width_ratio = kwargs['length_width_ratio']

    if 'add_on_ratio' not in kwargs.keys():
        add_on_ratio_width, add_on_ratio_length = 0.4, 0.6
    else:
        add_on_ratio_width, add_on_ratio_length = kwargs['add_on_ratio']

    if 'title' not in kwargs.keys():
        title = ''
    else:
        title = kwargs['title']

    if 'line2d_property' not in kwargs.keys():
        line2d_property = {'linewidth': 3}
    else:
        line2d_property = {**{'linewidth': 3}, **kwargs['line2d_property']}

    if 'polygon_property' not in kwargs.keys():
        polygon_property = {'zorder': 100}
    else:
        polygon_property = {**{'zorder': 100}, **kwargs['polygon_property']}

    if 'text_property' not in kwargs.keys():
        text_property = {'zorder': 100, 'ha': 'center', 'va': 'center'}
    else:
        text_property = {**{'zorder': 100, 'ha': 'center', 'va': 'center'}, **kwargs['text_property']}

    annotate = [] if 'annotate' not in kwargs.keys() else kwargs['annotate']

    tbu_lenth = 1
    tbu_width = tbu_lenth * length_width_ratio

    ax = plt.figure(figsize=[8, 8])
    ax.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')

    for tbu, value in circuit_element.items():
        # as an example, tbu = tbum1_1#2_1#3_v
        cell1_name, cell2_name = tbu.split("_")[1:3]
        cell1_center = cell_place_func(cell1_name, tbu_lenth, tbu_width)
        cell2_center = cell_place_func(cell2_name, tbu_lenth, tbu_width)

        # plot the background two edges, default color is mediumslateblue.
        edge1, edge2 = _get_two_edges(tbu_lenth, tbu_width, cell1_center, cell2_center)
        plt.plot(edge1[:, 0], edge1[:, 1], mcolors.CSS4_COLORS['mediumslateblue'], **line2d_property)
        plt.plot(edge2[:, 0], edge2[:, 1], mcolors.CSS4_COLORS['mediumslateblue'], **line2d_property)

        # plot the add-on rectangle, representing the phase shifts, default color is red.
        add_on_edge1, add_on_edge2 = _get_two_edges(tbu_lenth * (1 - add_on_ratio_length),
                                                    tbu_width * (1 + add_on_ratio_width), cell1_center, cell2_center)
        pts = np.concatenate((add_on_edge1, add_on_edge2), axis=0)
        hull = ConvexHull(pts)
        plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], 'red', **polygon_property)

        # deal with different annotation style
        if ('all_ps_symbol_value' in annotate or 'all_ps_value' in annotate) and (
                'theta' not in value.keys() or 'phi' not in value.keys()):
            raise RuntimeError("When 'all_ps_(symbol)_value' is used, phase shift values must be provided.")

        if 'all_ps_symbol_value' in annotate:
            text1_loc, text2_loc = add_on_edge1.mean(axis=0), add_on_edge2.mean(axis=0)
            theta_cell_name, phi_cell_name = value['ln'][0].split('_')[1], value['ln'][1].split('_')[1]
            plt.text(*text1_loc,
                     r'$\theta=%.1f\pi$' % (
                             value['theta'] / np.pi) if theta_cell_name == cell1_name else r'$\phi = %.1f\pi$' % (
                             value['phi'] / np.pi),
                     **text_property)
            plt.text(*text2_loc,
                     r'$\theta=%.1f\pi$' % (
                             value['theta'] / np.pi) if theta_cell_name == cell2_name else r'$\phi=%.1f\pi$' % (
                             value['phi'] / np.pi),
                     **text_property)

        if 'all_ps_symbol' in annotate:
            text1_loc, text2_loc = add_on_edge1.mean(axis=0), add_on_edge2.mean(axis=0)
            theta_cell_name, phi_cell_name = value['ln'][0].split('_')[1], value['ln'][1].split('_')[1]
            plt.text(*text1_loc,
                     r'$\theta$' if theta_cell_name == cell1_name else r'$\phi$',
                     **text_property)
            plt.text(*text2_loc,
                     r'$\theta$' if theta_cell_name == cell2_name else r'$\phi$',
                     **text_property)

        if 'all_ps_value' in annotate:
            text1_loc, text2_loc = add_on_edge1.mean(axis=0), add_on_edge2.mean(axis=0)
            theta_cell_name, phi_cell_name = value['ln'][0].split('_')[1], value['ln'][1].split('_')[1]
            plt.text(*text1_loc,
                     r'$%.1f\pi$' % (value['theta'] / np.pi) if theta_cell_name == cell1_name else r'$%.1f\pi$' % (
                             value['phi'] / np.pi),
                     **text_property)
            plt.text(*text2_loc,
                     r'$%.1f\pi$' % (value['theta'] / np.pi) if theta_cell_name == cell2_name else r'$%.1f\pi$' % (
                             value['phi'] / np.pi),
                     **text_property)

        if 'all_cell' in annotate:
            plt.text(*cell1_center, '(' + cell1_name.replace('#', ',') + ')', **text_property)
            plt.text(*cell2_center, '(' + cell2_name.replace('#', ',') + ')', **text_property)

        if 'all_tbu_name' in annotate:
            plt.text(*(cell1_center + cell2_center) / 2.0, tbu, **text_property)

    if title:
        plt.title(title)

    plt.show()


if __name__ == '__main__':
    circuit_element = generate('square_1', [3, 2], init_dict={'theta': 0, 'phi': 0.5 * np.pi})
    visualize(circuit_element, 'square_1', line2d_property={}, polygon_property={},
              annotate=['all_cell', 'all_ps_symbol'])

    circuit_element = generate('hexagonal_1', [2], init_dict={'theta': 0, 'phi': 0.5 * np.pi})
    visualize(circuit_element, 'hexagonal_1', line2d_property={}, polygon_property={},
              annotate=['all_cell', 'all_ps_symbol'])

    circuit_element = generate('triangular_1', [3, 5], init_dict={'theta': 0, 'phi': 0.5 * np.pi})
    visualize(circuit_element, 'triangular_1', line2d_property={}, polygon_property={},
              annotate=['all_cell', 'all_ps_symbol'], text_property={})
