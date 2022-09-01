import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Set, Tuple, Any, Callable, Type
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


def visualize(circuit_element: dict,
              placement: Optional[str] = '',
              cell_place_function: Callable = None,
              **kwargs) -> None:
    """Visualize a programmable photonics circuit containing TBUs

    :param circuit_element: A dict contains all TBUs and its connections
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

    :return: None.

    """

    if placement == '' and cell_place_function is None:
        raise RuntimeError("One of 'notation' and 'cell_place_function' should be provided to do visualization.")

    cell_place_function = cell_place_function if cell_place_function is not None else _cell_place_function_list(
        placement)
    _visualize_base(circuit_element, cell_place_function, **kwargs)


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
    a, b = direction_vector[1], -direction_vector[0]
    t = tbu_length / 2.0 / (a ** 2 + b ** 2) ** 0.5
    edge1 = np.array([point1 + t * np.array([a, b]), point1 - t * np.array([a, b])])
    edge2 = np.array([point2 + t * np.array([a, b]), point2 - t * np.array([a, b])])

    return edge1, edge2


def _visualize_base(circuit_element: dict, cell_place_func: Callable, node_signal: Optional[np.ndarray] = None,
                    node2ind: Optional[dict] = None, **kwargs):
    """"A base function for visualization

    :param circuit_element: A dictionary contains all TBUs and its connections
    :param node_signal: An np.ndarray of shape (#nodes, len(omega), 2) contains complex signals at each port.
    :param node2ind: A dict contains the mapping from node name to node index.
    :param cell_place_func: A callable function, given the cell name, length and width of TBU, return the position of
                            cell center.
    :param kwargs: a set of parameters determining the visualization. Accepted parameters including:

        length_width_ratio: a float represents the ratio of TBU length over TBU width
        add_on_ratio: a tuple with two floats, the first for length of add-on in the 'width' direction, the
                    second for length of add-on in the 'length' direction.
        title: the title of the figure.
        line2d_property: a dictionary with properties accepted by plt.plot()
        polygon_property: a dictionary with properties accepted by plt.fill()
        text_property: a dictionary with properties accepted by plt.text()
        annotate: a list determined what to display on the figure
    :return: None.

    """

    if 'length_width_ratio' not in kwargs.keys():
        length_width_ratio = 0.15
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

    freqeuncy_index = -1
    for cur_annotate in annotate:
        if isinstance(cur_annotate, dict):
            for key in cur_annotate.keys():
                if key == 'all_node':
                    node_annotate = cur_annotate['all_node']
                    freqeuncy_index = cur_annotate.get('frequency_index', 0)
                    filter_function = cur_annotate.get('filter_function', lambda x: np.abs(x) - 0.2 >= 0)
                    ratio2center = cur_annotate.get('ratio2center', 0.3)

    tbu_lenth = 2
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
            # theta_cell_name, phi_cell_name = value['ln'][0].split('_')[1], value['ln'][1].split('_')[1]
            plt.text(*text1_loc,
                     r'$\theta=%.2f\pi$' % (value['theta'] / np.pi),
                     **text_property)
            plt.text(*text2_loc,
                     r'$\phi=%.2f\pi$' % (value['phi'] / np.pi),
                     **text_property)

        if 'all_ps_symbol' in annotate:
            text1_loc, text2_loc = add_on_edge1.mean(axis=0), add_on_edge2.mean(axis=0)
            plt.text(*text1_loc, r'$\theta$', **text_property)
            plt.text(*text2_loc, r'$\phi$', **text_property)

        if 'all_ps_value' in annotate:
            text1_loc, text2_loc = add_on_edge1.mean(axis=0), add_on_edge2.mean(axis=0)
            plt.text(*text1_loc,
                     r'$%.2f\pi$' % (value['theta'] / np.pi),
                     **text_property)
            plt.text(*text2_loc,
                     r'$%.2f\pi$' % (value['phi'] / np.pi),
                     **text_property)

        if 'all_cell' in annotate:
            plt.text(*cell1_center, '(' + cell1_name.replace('#', ',') + ')', **text_property)
            plt.text(*cell2_center, '(' + cell2_name.replace('#', ',') + ')', **text_property)

        if 'all_tbu_name' in annotate:
            plt.text(*(cell1_center + cell2_center) / 2.0, tbu, **text_property)

        def _convert(src_value):
            if node_annotate == 'real':
                return np.real(src_value)
            if node_annotate == 'imag':
                return np.imag(src_value)
            if node_annotate == 'angle':
                return np.angle(src_value)
            if node_annotate == 'abs':
                return np.abs(src_value)
            if node_annotate == 'complex':
                return src_value

        def _annotate_node(node_name):
            index = node2ind[node_name]
            complex_response = node_signal[freqeuncy_index, [2 * index, 2 * index + 1], 0]
            ind = np.argmax(np.abs(complex_response))
            text = _convert(complex_response[ind])
            # TODO: need some thinking, visualize what? since there are two magnitudes at one port.
            if node_annotate == 'real':
                return "{:.2f}".format(text) if filter_function(complex_response[ind]) else ""
            if node_annotate == 'imag':
                return "{:.2f}j".format(text) if filter_function(complex_response[ind]) else ""
            if node_annotate == 'angle':
                return "{:.1f}pi".format(text / np.pi) if filter_function(complex_response[ind]) else ""
            if node_annotate == 'abs':
                return "{:.2f}".format(text) if filter_function(complex_response[ind]) else ""
            if node_annotate == 'complex':
                return "{:.2f}".format(text) if filter_function(complex_response[ind]) else ""
            if node_annotate == 'name':
                return node_name if filter_function(complex_response[ind]) else ""

        if freqeuncy_index > -1:
            if node_signal is None or node2ind is None:
                raise RuntimeError("'node_signal' and 'node2index' must be provided if annotating node response.")
            if freqeuncy_index > node_signal.shape[1]:
                raise RuntimeError(
                    "The frequency index %d is not in the range of [0, %d)" % (freqeuncy_index, node_signal.shape[1]))

            plt.text(*((1 - ratio2center) * edge1[0] + ratio2center * cell1_center), _annotate_node(value['ln'][0]),
                     **text_property)
            plt.text(*((1 - ratio2center) * edge2[0] + ratio2center * cell2_center), _annotate_node(value['ln'][1]),
                     **text_property)

            plt.text(*((1 - ratio2center) * edge1[1] + ratio2center * cell1_center), _annotate_node(value['rn'][0]),
                     **text_property)
            plt.text(*((1 - ratio2center) * edge2[1] + ratio2center * cell2_center), _annotate_node(value['rn'][1]),
                     **text_property)

    if title:
        plt.title(title)

    plt.show()
