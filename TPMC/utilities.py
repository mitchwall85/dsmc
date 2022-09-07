import numpy as np

def reflect_spectular(c: np.array, wall: np.array):
    """calculate the reflected velocity for a specular wall impact

    Args:
        c (np.array): incomming velocity
        wall (np.array): wall normal vector
    """
    # ensure wall vector is a unit vector
    wall = wall/np.linalg.norm(wall)

    c_n = np.dot(c,wall)*wall # normal component to wall
    c_p = c - c_n # perpendicular component to wall
    c_prime = c_p - c_n # flip normal component
    return c_prime
