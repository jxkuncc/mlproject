import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import newton

def apply_htm(htm:np.ndarray, A:np.ndarray):
    """
        Applies the htm to the array A
    Args:
        htm (np.ndarray): _description_
        A (np.ndarray): _description_

    Returns:
        _type_: A transformed by the HTM
    """
    assert A.shape[1] == 3
    # Rotate the matrix so that it's in vector form
    bf = np.rot90(A, axes=(1,0))
    # add a row of ones to apply the htm
    bf = np.vstack((bf, np.ones((1, bf.shape[1]))))
    # apply the htm
    bf = np.matmul(htm, bf)
    #Drop the extra ones
    bf = bf[:-1]
    #Rotate it back to x,y,z form
    return np.rot90(bf, axes=(0,1))    

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def _weighted_fit(x:np.ndarray, points, expected_values, stiffness):
    
    translation = x[:3]
    rot = Rotation.from_euler('xyz', x[3:])
    points = translation + rot.apply(points)

    # Sum of the Forces
    sum_x_f = np.sum(stiffness[:,0] * (expected_values[:,0] - points[:,0]))
    sum_y_f = np.sum(stiffness[:,1] * (expected_values[:,1] - points[:,1]))
    sum_z_f = np.sum(stiffness[:,2] * (expected_values[:,2] - points[:,2]))
    # Sum of the Moments
    # TODO Moments around centroid?
    moments = np.sum(np.cross(points, (expected_values-points)*stiffness), axis = 0)
    out = np.array((sum_x_f, sum_y_f, sum_z_f, moments[0], moments[1], moments[2]))
    return out


def weighted_fit(A, B, spread):
    "fit A to B give the variability in A"

    stiffness = np.divide(np.ones_like(spread), spread)

    _, guess_rot, guess_trans = best_fit_transform(A, B)
    guess = np.hstack((guess_trans, Rotation.from_matrix(guess_rot).as_euler('xyz')))

    x0 = newton(_weighted_fit, guess, args=(A, B, stiffness), tol=10e-010, maxiter=500)

    translation = x0[:3]
    rotation = Rotation.from_euler('xyz', x0[3:]).as_matrix()

    return (translation, rotation)