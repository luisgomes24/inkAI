import numpy as np


def get_box_center(box):
    # Box: [x1,y1,x2,y2]
    [x1, y1, x2, y2] = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def sort_coordinates_x(coords):
    # Return the indexes of the sorted coordinates
    coords = np.array(coords, ndmin=2)

    return np.argsort(coords[:, 0])


def get_nearest_coord(box_center, box_centers_list):
    # Return the index of the nearest box
    box_centers_array = np.asarray(box_centers_list)
    dist_2 = np.sum((box_centers_array - box_center) ** 2, axis=1)
    return np.argmin(dist_2)
