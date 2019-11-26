import numpy as np
import scipy.linalg  # you may find scipy.linalg.block_diag useful
import turtlebot_model as tb


class Ekf(object):
    """
    Base class for EKF Localization and SLAM.

    Usage:
        ekf = EKF(x0, Sigma0, R)
        while True:
            ekf.transition_update(u, dt)
            ekf.measurement_update(z, Q)
            localized_state = ekf.x
    """

    def __init__(self, x0, Sigma0, R):
        """
        EKF constructor.

        Inputs:
                x0: np.array[n,]  - initial belief mean.
            Sigma0: np.array[n,n] - initial belief covariance.
                 R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.x = x0  # Gaussian belief mean
        self.Sigma = Sigma0  # Gaussian belief covariance
        self.R = R  # Control noise covariance (corresponding to dt = 1 second)

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating (self.x, self.Sigma).

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        g, Gx, Gu = self.transition_model(u, dt)

        ########## Code starts here ##########
        # TODO: Update self.x, self.Sigma.
        self.x = g  # np.matmul(Gx,self.x) + np.matmul(Gu, u)

        self.Sigma = np.matmul(Gx, np.matmul(self.Sigma, np.transpose(Gx))) + dt * np.matmul(Gu, np.matmul(self.R,
                                                                                                           np.transpose(
                                                                                                               Gu)))
        g

        ########## Code ends here ##########

    def transition_model(self, u, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
             u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Outputs:
             g: np.array[n,]  - result of belief mean propagated according to the
                                system dynamics with control u for dt seconds.
            Gx: np.array[n,n] - Jacobian of g with respect to belief mean self.x.
            Gu: np.array[n,2] - Jacobian of g with respect to control u.
        """
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.Sigma) should be updated.
        """
        z, Q, H = self.measurement_model(z_raw, Q_raw)
        if z is None:
            # Don't update if measurement is invalid
            # (e.g., no line matches for line-based EKF localization)
            return

        ########## Code starts here ##########
        # TODO: Update self.x, self.Sigma.

        S = np.matmul(H, np.matmul(self.Sigma, np.transpose(H))) + Q
        K = np.matmul(self.Sigma, np.matmul(np.transpose(H), np.linalg.inv(S)))

        self.x = self.x + np.matmul(K, z).reshape(-1)
        self.Sigma = self.Sigma - np.matmul(K, np.matmul(S, np.transpose(K)))
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction). Also returns the associated Jacobian for EKF
        linearization.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2K,]   - measurement mean.
            Q: np.array[2K,2K] - measurement covariance.
            H: np.array[2K,n]  - Jacobian of z with respect to the belief mean self.x.
        """
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class EkfLocalization(Ekf):
    """
    EKF Localization.
    """

    def __init__(self, x0, Sigma0, R, map_lines, tf_base_to_camera, g):
        """
        EkfLocalization constructor.

        Inputs:
                       x0: np.array[3,]  - initial belief mean.
                   Sigma0: np.array[3,3] - initial belief covariance.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Turtlebot dynamics (unicycle model).
        """

        ########## Code starts here ##########
        # TODO: Compute g, Gx, Gu using tb.compute_dynamics().

        g, Gx, Gu = tb.compute_dynamics(self.x, u, dt)
        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature.
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print("Scanner sees {} lines but can't associate them with any map entries."
                  .format(z_raw.shape[1]))
            return None, None, None

        ########## Code starts here ##########
        # TODO: Compute z, Q.
        vertical = True

        if v_list[0].shape != (2, 1):
            v_list[0] = v_list[0].reshape(2, 1)
            vertical = False

        z = v_list[0]
        H = H_list[0]
        Q = Q_list[0]

        for k in range(len(v_list)):
            if not vertical:
                v_list[k] = v_list[k].reshape(2, 1)

            z = np.vstack((z, v_list[k]))
            H = np.vstack((H, H_list[k]))
            Q = scipy.linalg.block_diag(Q, Q_list[k])

        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            v_list: [np.array[2,]]  - list of at most I innovation vectors
                                      (predicted map measurement - scanner measurement).
            Q_list: [np.array[2,2]] - list of covariance matrices of the
                                      innovation vectors (from scanner uncertainty).
            H_list: [np.array[2,3]] - list of Jacobians of the innovation
                                      vectors with respect to the belief mean self.x.
        """

        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########
        # TODO: Compute v_list, Q_list, H_list
        num_predictions = len(Hs)
        num_measurements = len(Q_raw)

        d = np.zeros((num_measurements, num_predictions))
        v = np.zeros((2, num_measurements, num_predictions))

        v_list = []
        Q_list = []
        H_list = []

        for i in range(num_measurements):
            for j in range(num_predictions):
                v[:, i, j] = z_raw[:, i] - hs[:, j]

                S = np.matmul(Hs[j], np.matmul(self.Sigma, np.transpose(Hs[j]))) + Q_raw[i]

                d[i, j] = np.matmul(np.transpose(v[:, i, j]), np.matmul(np.linalg.inv(S), v[:, i, j]))

            j_min = np.argmin(d[i, :])

            if d[i, j_min] < self.g ** 2:
                v_list.append(v[:, i, j_min])
                Q_list.append(Q_raw[i])
                H_list.append(Hs[j_min])

        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Outputs:
                 hs: np.array[2,J]  - J line parameters in the scanner (camera) frame.
            Hx_list: [np.array[2,3]] - list of Jacobians of h with respect to the belief mean self.x.
        """
        hs = np.zeros_like(self.map_lines)
        Hx_list = []
        for j in range(self.map_lines.shape[1]):
            ########## Code starts here ##########
            # TODO: Compute h, Hx using tb.transform_line_to_scanner_frame().
            h, Hx = tb.transform_line_to_scanner_frame(self.map_lines[:, j], self.x, self.tf_base_to_camera)

            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[:, j] = h
            Hx_list.append(Hx)

        return hs, Hx_list


class EkfSlam(Ekf):
    """
    EKF SLAM.
    """

    def __init__(self, x0, Sigma0, R, tf_base_to_camera, g):
        """
        EKFSLAM constructor.

        Inputs:
                       x0: np.array[3+2J,]     - initial belief mean.
                   Sigma0: np.array[3+2J,3+2J] - initial belief covariance.
                        R: np.array[2,2]       - control noise covariance
                                                 (corresponding to dt = 1 second).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, Sigma0, R)

    def transition_model(self, u, dt):
        """
        Combined Turtlebot + map dynamics.
        Adapt this method from EkfLocalization.transition_model().
        """
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))

        ########## Code starts here ##########
        # TODO: Compute g, Gx, Gu.

        xt_1 = g[0]
        yt_1 = g[1]
        tht_1 = g[2]

        V = u[0]
        om = u[1]

        EPSILON_OMEGA = 0.001

        if np.abs(u[1]) > EPSILON_OMEGA:
            tht = om * dt + tht_1  # theta
            g[2] = tht
            g[0] = xt_1 + V / om * (np.sin(tht) - np.sin(tht_1))  # x
            g[1] = yt_1 + V / om * (np.cos(tht_1) - np.cos(tht))  # y

            Gx[0, 2] = V / om * (np.cos(tht) - np.cos(tht_1))  # del_th/del_x
            Gx[1, 2] = V / om * (np.sin(tht) - np.sin(tht_1))  # del_th/del_y

            Gu[0, 0] = 1.0 / om * (np.sin(tht) - np.sin(tht_1))
            Gu[0, 1] = V / np.power(om, 2) * (np.sin(tht_1) - np.sin(tht)) + V * dt / om * np.cos(tht)

            Gu[1, 0] = 1.0 / om * (np.cos(tht_1) - np.cos(tht))
            Gu[1, 1] = V / np.power(om, 2) * (np.cos(tht) - np.cos(tht_1)) + V * dt / om * np.sin(tht)

            Gu[2, 1] = dt

        else:
            tht = om * dt + tht_1
            g[2] = tht
            g[0] = xt_1 + V * dt * np.cos(tht)
            g[1] = yt_1 + V * dt * np.sin(tht)

            Gx[0, 2] = -V * dt * np.sin(tht)  # del_th/del_x
            Gx[1, 2] = V * dt * np.cos(tht)  # del_th/del_y

            Gu[0, 0] = dt * np.cos(tht)  # del_x/del_V
            Gu[0, 1] = -V * dt ** 2 * np.sin(tht)  # del_x/del_om

            Gu[1, 0] = dt * np.sin(tht)  # del_y/del_V
            Gu[1, 1] = V * dt ** 2 * np.cos(tht)  # del_y/del_om

            Gu[2, 1] = dt  # del_th/del_om
        ########## Code ends here ##########

        return g, Gx, Gu

    def measurement_model(self, z_raw, Q_raw):
        """
        Combined Turtlebot + map measurement model.
        Adapt this method from EkfLocalization.measurement_model().

        The ingredients for this model should look very similar to those for
        EkfLocalization. In particular, essentially the only thing that needs to
        change is the computation of Hx in self.compute_predicted_measurements()
        and how that method is called in self.compute_innovations() (i.e.,
        instead of getting world-frame line parameters from self.map_lines, you
        must extract them from the state self.x).
        """
        v_list, Q_list, H_list = self.compute_innovations(z_raw, Q_raw)
        if not v_list:
            print("Scanner sees {} lines but can't associate them with any map entries."
                  .format(z_raw.shape[1]))
            return None, None, None

        ########## Code starts here ##########
        # TODO: Compute z, Q, H.
        # Hint: Should be identical to EkfLocalization.measurement_model().

        vertical = True

        if v_list[0].shape != (2, 1):
            v_list[0] = v_list[0].reshape(2, 1)
            vertical = False

        z = v_list[0]
        H = H_list[0]
        Q = Q_list[0]

        for k in range(len(v_list)):
            if not vertical:
                v_list[k] = v_list[k].reshape(2, 1)

            z = np.vstack((z, v_list[k]))
            H = np.vstack((H, H_list[k]))
            Q = scipy.linalg.block_diag(Q, Q_list[k])
        ########## Code ends here ##########

        return z, Q, H

    def compute_innovations(self, z_raw, Q_raw):
        """
        Adapt this method from EkfLocalization.compute_innovations().
        """

        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        hs, Hs = self.compute_predicted_measurements()

        ########## Code starts here ##########
        # TODO: Compute v_list, Q_list, H_list.
        num_predictions = len(Hs)
        num_measurements = len(Q_raw)

        d = np.zeros((num_measurements, num_predictions))
        v = np.zeros((2, num_measurements, num_predictions))

        v_list = []
        Q_list = []
        H_list = []

        for i in range(num_measurements):
            for j in range(num_predictions):
                v[:, i, j] = z_raw[:, i] - hs[:, j]

                S = np.matmul(Hs[j], np.matmul(self.Sigma, np.transpose(Hs[j]))) + Q_raw[i]

                d[i, j] = np.matmul(np.transpose(v[:, i, j]), np.matmul(np.linalg.inv(S), v[:, i, j]))

            j_min = np.argmin(d[i, :])

            if d[i, j_min] < self.g ** 2:
                v_list.append(v[:, i, j_min])
                Q_list.append(Q_raw[i])
                H_list.append(Hs[j_min])
        ########## Code ends here ##########

        return v_list, Q_list, H_list

    def compute_predicted_measurements(self):
        """
        Adapt this method from EkfLocalization.compute_predicted_measurements().
        """
        J = (self.x.size - 3) // 2
        hs = np.zeros((2, J))
        Hx_list = []
        for j in range(J):
            idx_j = 3 + 2 * j
            alpha, r = self.x[idx_j:idx_j + 2]

            Hx = np.zeros((2, self.x.size))

            ########## Code starts here ##########
            # TODO: Compute h, Hx.

            x = self.x[0:3]

            rotation = np.array([
                [np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]
            ])

            base_to_camera_xy = np.array([
                [self.tf_base_to_camera[0]],
                [self.tf_base_to_camera[1]]
            ])

            base_to_camera_xy_prime = np.matmul(rotation, base_to_camera_xy)

            world_to_camera = base_to_camera_xy_prime + np.array([[x[0]], [x[1]]])

            r_c = r - np.cos(alpha) * world_to_camera[0, 0] - np.sin(alpha) * world_to_camera[1, 0]
            alpha_c = alpha - self.tf_base_to_camera[2] - x[2]

            h = np.array([alpha_c, r_c])

            method = 2
            if method == 1:
                q = (r*np.cos(alpha) - x[0])**2 + (r*np.sin(alpha) - x[1])**2

                a = (x[0] - r*np.cos(alpha))/np.sqrt(q)
                b = (x[1] - r*np.sin(alpha))/np.sqrt(q)
                c = -b/np.sqrt(q)
                d = a/np.sqrt(q)

                Hx[0, 0] = a
                Hx[0, 1] = b
                Hx[1, 0] = c
                Hx[1, 1] = d
                Hx[1, 2] = -1
            elif method == 2:
                del_r_del_th = self.tf_base_to_camera[0] * np.cos(alpha) * np.sin(x[2]) + self.tf_base_to_camera[1] * \
                    np.cos(alpha) * np.cos(x[2]) - self.tf_base_to_camera[0] * np.sin(alpha) * np.cos(x[2]) + \
                    self.tf_base_to_camera[1] * np.sin(alpha) * np.sin(x[2])

                del_r_del_alpha = np.sin(alpha) * (
                    self.tf_base_to_camera[0] * np.cos(x[2]) - self.tf_base_to_camera[1] * np.sin(x[2]) + x[0]
                ) - np.cos(alpha) * (
                    self.tf_base_to_camera[0] * np.sin(x[2]) + self.tf_base_to_camera[1] * np.cos(x[2]) + x[1]
                )

                Hx[0, 2] = -1
            # Hx[0, idx_j] = 1

                Hx[1, 0] = -np.cos(alpha)
                Hx[1, 1] = -np.sin(alpha)
                Hx[1, 2] = del_r_del_th
            else:
                print("WRONG METHOD NUMBER")
                return -1

            # Hx[1, idx_j] = del_r_del_alpha
            # Hx[1, idx_j + 1] = 1

            # First two map lines are assumed fixed so we don't want to propagate
            # any measurement correction to them.
            if j >= 2:
                if method == 1:
                    Hx[:, idx_j:idx_j + 2] = np.array([
                        [-a, -b],
                        [-c, -d]
                    ])  # FIX ME!
                else:
                    Hx[:, idx_j:idx_j + 2] = np.eye(2)
                    Hx[1, idx_j] = del_r_del_alpha      # FIXED!!
            ########## Code ends here ##########

            h, Hx = tb.normalize_line_parameters(h, Hx)
            hs[:, j] = h
            Hx_list.append(Hx)

        return hs, Hx_list
