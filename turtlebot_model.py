import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(x, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                        x: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to x.
        Gu: np.array[3,2] - Jacobian of g with respect ot u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    xt_1 = x[0]
    yt_1 = x[1]
    tht_1 = x[2]

    V = u[0]
    om = u[1]

    if np.abs(u[1]) > EPSILON_OMEGA:
        tht = om*dt + tht_1
        xt = xt_1 + V/om*(np.sin(tht) - np.sin(tht_1))
        yt = yt_1 + V/om*(np.cos(tht_1) - np.cos(tht))

        g = np.array([xt, yt, tht])

        if compute_jacobians:
            Gx = np.array([
            [1, 0, V/om*(np.cos(tht) - np.cos(tht_1))],
            [0, 1, V/om*(np.sin(tht) - np.sin(tht_1))],
            [0, 0, 1]
            ])

            Gu = np.array([
            [1.0/om*(np.sin(tht) - np.sin(tht_1)), V/np.power(om, 2)*(np.sin(tht_1) - np.sin(tht)) + V*dt/om*np.cos(tht)],
            [1.0/om*(np.cos(tht_1) - np.cos(tht)), V/np.power(om, 2)*(np.cos(tht) - np.cos(tht_1)) + V*dt/om*np.sin(tht)],
            [0, dt]
            ])
    else:
        tht = om*dt + tht_1
        xt = xt_1 + V*dt*np.cos(tht)
        yt = yt_1 + V*dt*np.sin(tht)

        g = np.array([xt, yt, tht])

        if compute_jacobians:
            Gx = np.array([
            [1, 0, -V*dt*np.sin(tht)],
            [0, 1, V*dt*np.cos(tht)],
            [0, 0, 1]
            ])

            Gu = np.array([
            [dt*np.cos(tht), -V*dt**2*np.sin(tht)],
            [dt*np.sin(tht), V*dt**2*np.cos(tht)],
            [0, dt]
            ])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    rotation = np.array([
    [np.cos(x[2]), -np.sin(x[2])],
    [np.sin(x[2]), np.cos(x[2])]
    ])

    base_to_camera_xy = np.array([
    [tf_base_to_camera[0]],
    [tf_base_to_camera[1]]
    ])

    base_to_camera_xy_prime = np.matmul(rotation, base_to_camera_xy)

    world_to_camera = base_to_camera_xy_prime + np.array([ [x[0]], [x[1]] ])

    r_c = r - np.cos(alpha)*world_to_camera[0, 0] - np.sin(alpha)*world_to_camera[1, 0]
    alpha_c = alpha - tf_base_to_camera[2] - x[2]

    h = np.array([alpha_c, r_c])

    if compute_jacobian:
        temp = tf_base_to_camera[0]*np.cos(alpha)*np.sin(x[2]) + tf_base_to_camera[1]*np.cos(alpha)*np.cos(x[2]) - tf_base_to_camera[0]*np.sin(alpha)*np.cos(x[2]) + tf_base_to_camera[1]*np.sin(alpha)*np.sin(x[2])

        Hx = np.array([
        [0, 0, -1],
        [-np.cos(alpha), -np.sin(alpha), temp]
        ])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
