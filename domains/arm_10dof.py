import numpy as np

# # # Forward Kinematics (2D Chain) # # #

def forward_kinematics(joint_angles, link_lengths):
    """
    Computes 2D end-effector position of a kinematic chain.
    joint_angles: shape (10,)
    link_lengths: shape (10,)
    Returned: (x, y)
    """
    x, y = 0.0, 0.0
    theta = 0.0

    for ang, L in zip(joint_angles, link_lengths):
        theta += ang
        x += L * np.cos(theta)
        y += L * np.sin(theta)

    return np.array([x, y])


# # # Fitness function # # #

def fitness_arm(x, theta):
    """
    Implements the 10-DoF Arm fitness used in the PT-ME paper.

    x     = solution vector in [0,1]^10  (normalized joint angles)
    theta = task parameter vector of size 2:
            theta[0] = max joint angle (scaled to e.g. pi)
            theta[1] = link length scale factor
    """

    # Task parameter unpack
    max_angle = theta[0] * np.pi           # maximum joint angle (0 to pi)
    link_scale = 0.5 + theta[1] * 1.5      # lengths vary from 0.5 to 2.0

    # Construct joint angles + link lengths
    joint_angles = x * max_angle           # scale normalized angles
    link_lengths = np.ones(10) * link_scale

    # Compute end-effector pos
    ee_pos = forward_kinematics(joint_angles, link_lengths)

    # Fixed target (like Fig 2 from paper)
    target = np.array([5.0, 0.0])

    # Euclidean distance
    dist = np.linalg.norm(ee_pos - target)

    # Fitness = exp(-dist^2), bounded 0–1
    f = np.exp(-dist**2)

    return float(f)
