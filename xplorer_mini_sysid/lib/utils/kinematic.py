import numpy as np

def ssa(desired, actual):
    err = desired - actual
    if err > np.pi:
        err = err - (2.0 * np.pi)
    elif err < -np.pi:
        err = err + (2.0 * np.pi)
    
    return err

def cal_eta_err_with_ssa(eta_ref, eta):
    """
    Calculate the error between the current pose and the reference pose.
    
    Args:
        eta (np.array): Current pose (6D vector).
        eta_ref (np.array): Reference pose (6D vector).

    Returns:
        np.array: Error between current and reference pose (6D vector).

    """

    # Ensure inputs are numpy arrays
    eta = np.array(eta)
    eta_ref = np.array(eta_ref)

    # Calculate error using SSA for orientation components
    error = np.zeros_like(eta)
    error[:3] = eta_ref[:3] - eta[:3]  # Position error

    for j in range(3):
        error[3+j] = ssa(eta_ref[3+j], eta[3+j])  # Orientation error using SSA

    return error

def eulerang(phi, theta, psi):
        """  
        Generate the transformation 6x6 matrix J 
        and 3x3 matrix of j_11 and j_22
        which corresponds to eq 2.40 on p.26 (Fossen 2011)
        """
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth  = np.cos(theta)
        sth  = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        if cth==0: 
            return -1

        # corresponds to eq 2.18 on p.22 (Fossen 2011)
        r_zyx = np.array([[cpsi*cth,  -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth],
                [spsi*cth,  cpsi*cphi+sphi*sth*spsi,   -cpsi*sphi+sth*spsi*cphi],
                [-sth,      cth*sphi,                  cth*cphi ]])

        # corresponds to eq 2.28 on p.25 (Fossen 2011)
        t_zyx = np.array([[1,  sphi*sth/cth,  cphi*sth/cth],
            [0,  cphi,          -sphi],
            [0,  sphi/cth,      cphi/cth]])

        # corresponds to eq 2.40 on p.26 (Fossen 2011)
        j_1 = np.concatenate((r_zyx, np.zeros((3,3))), axis=1)
        j_2 = np.concatenate((np.zeros((3,3)), t_zyx), axis=1)
        j = np.concatenate((j_1, j_2), axis=0)

        return j, r_zyx, t_zyx