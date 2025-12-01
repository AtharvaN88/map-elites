import numpy as np


def simulate_projectile(yaw, pitch, distance, wind, g=9.81, v0=30.0):
    """
    Simulates a simple 2D projectile model with:
    - yaw angle in radians (horizontal deviation)
    - pitch angle in radians (elevation)
    - distance to target
    - wind as a constant horizontal acceleration
    
    Returns: (dx, dy) displacement of arrow from target center.
    """

    # Decompose firing angle into planar components
    vx0 = v0 * np.cos(pitch) * np.cos(yaw)
    vy0 = v0 * np.sin(pitch)
    vh0 = v0 * np.cos(pitch) * np.sin(yaw)  # horizontal sideways drift

    # Time to reach the target plane (x = distance)
    # distance = vx0 * t => t = distance / vx0
    if vx0 <= 0.01:
        return np.array([999, 999])  # invalid shot

    t = distance / vx0

    # Horizontal sideways drift (yaw + wind)
    # y_wind = 0.5 * wind * t^2
    side_disp = vh0 * t + 0.5 * wind * (t ** 2)

    # Vertical drop
    # y_vert(t) = vy0 * t - 0.5 g t^2
    vert_disp = vy0 * t - 0.5 * g * (t ** 2)

    # Need the impact relative to target center: (horizontal, vertical)
    return np.array([side_disp, vert_disp])


def fitness_archery(x, theta):
    """
    Archery task used in PT-ME paper.

    x: solution vector in [0,1]^2: yaw, pitch
    theta: task parameter [distance, wind].
           Both assumed to be in [0,1].

    Mapping:
      distance in [20m, 60m]
      wind     in [-5, +5] m/s^2 lateral acceleration

      yaw, pitch map to [-pi/6, pi/6]
    """

    # Unpack task
    distance = 20 + theta[0] * 40        # 20–60 meters
    wind = -5 + theta[1] * 10            # -5 to +5 m/s^2

    # Map solution space
    yaw_range = np.pi / 6
    pitch_range = np.pi / 6

    yaw = (x[0] * 2 - 1) * yaw_range      # [-pi/6, +pi/6]
    pitch = (x[1] * 2 - 1) * pitch_range  # [-pi/6, +pi/6]

    # Simulate physics
    impact = simulate_projectile(yaw, pitch, distance, wind)

    # Distance from bullseye
    miss = np.linalg.norm(impact)

    # Fitness = exp(-miss^2), same form as Arm
    return float(np.exp(-miss ** 2))
