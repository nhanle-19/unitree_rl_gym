import numpy as np


ANKLE_GAIN = {
    "grav_x": 0.0,
    "kp_x": 35.0,
    "kd_x": 13.0,
    "grav_y": 0.0,
    "kp_y": 25.0,
    "kd_y": 10.0,
}


def solve_discrete_are_iterative(a, b, q, r, max_iter=500, tol=1e-10):
    """Small NumPy-only DARE solver for the 2-state ACC regulators."""
    x = np.array(q, dtype=float)
    r = np.atleast_2d(np.array(r, dtype=float))
    for _ in range(max_iter):
        gain_denom = r + b.T @ x @ b
        x_next = a.T @ x @ a - a.T @ x @ b @ np.linalg.solve(gain_denom, b.T @ x @ a) + q
        if np.max(np.abs(x_next - x)) < tol:
            return x_next
        x = x_next
    return x


class AccHighController:
    """ACC high-level controller used to generate adaptive stance ankle torque."""

    def __init__(
        self,
        CtrlFreq=500,
        z_sc_d_set=0.80,
        T_step_set=0.5,
        vx_des_set=0.2,
        vy_des_set=0.0,
        width_des_set=0.15,
        mass_set=47.2281,
    ):
        self.CtrlFreq = CtrlFreq
        self.g = 9.81
        self.z_sc_d = z_sc_d_set
        self.mass = mass_set
        self.T_step = T_step_set
        self.vx_des = vx_des_set
        self.vy_des = vy_des_set
        self.width_des = width_des_set
        self.nom_step_size = self.vx_des * self.T_step

        self.config_step_ctrl()
        self.desired_coeff()
        self.config_step_ctrl_y()
        self.desired_coeff_y()

    def config_step_ctrl(self):
        g = self.g
        z = self.z_sc_d
        t_step = self.T_step
        l_step = np.sqrt(g / z)

        m_mat = np.array([[1, 1], [l_step, -l_step]])
        a_mat = np.diag([np.exp(l_step * t_step), np.exp(-l_step * t_step)])
        b_mat = -0.5 * np.array([[l_step, 1], [l_step, -1]]) / l_step
        c_mat = m_mat @ (a_mat - np.eye(2))

        inv_m = np.linalg.inv(m_mat)
        a_dare = m_mat @ a_mat @ inv_m
        b_dare = -(m_mat @ (a_mat - np.eye(2)) @ inv_m) @ np.array([[1], [0]])
        q_dare = np.diag([1, 1])
        r_dare = np.array([[1e-1]])

        x = solve_discrete_are_iterative(a_dare, b_dare, q_dare, r_dare)
        k_mat = np.linalg.solve(b_dare.T @ x @ b_dare + r_dare, b_dare.T @ x @ a_dare)

        self.fstep = 0.7
        self.Astep = a_mat
        self.Bstep = b_mat
        self.Cstep = c_mat
        self.Kstep = k_mat
        self.Lstep = l_step
        return self.fstep, a_mat, b_mat, c_mat, k_mat, l_step

    def desired_coeff(self):
        self.des_coeff = np.linalg.inv(self.Cstep) @ np.array([[self.nom_step_size], [0]])
        return self.des_coeff

    def solve_for_coeffs_from_ic(self, x, v, t_min_tk):
        l_step = self.Lstep
        m_mat = 0.5 * np.array([[l_step, 1], [l_step, -1]]) / l_step
        d_mat = np.diag([np.exp(-l_step * t_min_tk), np.exp(l_step * t_min_tk)])
        return d_mat @ m_mat @ np.array([[x], [v]])

    def update_commanded_profile_coeffs(self, curr_coef):
        outputs_to_be_reg = self.Cstep @ curr_coef - np.array([[self.nom_step_size], [0]])
        goal_step = float(np.asarray(self.nom_step_size - self.Kstep @ outputs_to_be_reg).squeeze())
        goal_coeff = self.Astep @ curr_coef + self.Bstep @ np.array([[goal_step], [0]], dtype=float)
        return goal_coeff, goal_step

    def config_step_ctrl_y(self):
        l_step = np.sqrt(self.g / self.z_sc_d)
        t_step = self.T_step

        m_mat = np.array([[1, 1], [l_step, -l_step]])
        exp_lt = np.exp(l_step * t_step)
        expm_lt = np.exp(-l_step * t_step)
        e_mat = np.array([[exp_lt, 0], [0, expm_lt]])
        g_mat = np.array([[exp_lt - 1, expm_lt - 1], [exp_lt + 1, expm_lt + 1]])

        a_dare_y = g_mat @ e_mat @ np.linalg.inv(g_mat)
        b_dare_y = -g_mat @ np.linalg.inv(m_mat) @ np.array([[1], [0]])
        q_dare_y = np.diag([100, 100])
        r_dare_y = np.array([[0.1]])

        x = solve_discrete_are_iterative(a_dare_y, b_dare_y, q_dare_y, r_dare_y)
        k_mat = np.linalg.solve(b_dare_y.T @ x @ b_dare_y + r_dare_y, b_dare_y.T @ x @ a_dare_y)

        self.Mstep_y = m_mat
        self.Estep_y = e_mat
        self.Gstep_y = g_mat
        self.Kstep_y = k_mat
        return m_mat, e_mat, g_mat, k_mat

    def desired_coeff_y(self):
        hy1_d = self.T_step * self.vy_des
        hy2_d_l = -2 * self.width_des
        hy2_d_r = 2 * self.width_des

        self.des_coeff_y_L = np.linalg.inv(self.Gstep_y) @ np.array([[hy1_d], [hy2_d_l]])
        self.des_coeff_y_R = np.linalg.inv(self.Gstep_y) @ np.array([[hy1_d], [hy2_d_r]])
        self.hy1_d = hy1_d
        self.hy2_d_L = hy2_d_l
        self.hy2_d_R = hy2_d_r
        return self.des_coeff_y_L, self.des_coeff_y_R

    def solve_for_coeffs_from_ic_y(self, y, vy, t_min_tk):
        l_step = self.Lstep
        ic = np.array(
            [
                [np.exp(l_step * t_min_tk), np.exp(-l_step * t_min_tk)],
                [l_step * np.exp(l_step * t_min_tk), -l_step * np.exp(-l_step * t_min_tk)],
            ]
        )
        return np.linalg.inv(ic) @ np.array([[y], [vy]])

    def update_commanded_profile_coeffs_y(self, i_stance_cur, curr_coeff_y):
        if i_stance_cur == 1:
            yd_k = np.array([[self.hy1_d], [self.hy2_d_L]])
            yd_kp1 = np.array([[self.hy1_d], [self.hy2_d_R]])
        elif i_stance_cur == 2:
            yd_k = np.array([[self.hy1_d], [self.hy2_d_R]])
            yd_kp1 = np.array([[self.hy1_d], [self.hy2_d_L]])
        else:
            raise ValueError("i_stance_cur must be 1 for left stance or 2 for right stance")

        ey_k = self.Gstep_y @ curr_coeff_y - yd_k
        aa = self.Gstep_y @ np.linalg.inv(self.Mstep_y)
        bb = self.Gstep_y @ self.Estep_y @ np.linalg.inv(self.Gstep_y) @ yd_k - yd_kp1
        u_kp1 = -np.linalg.inv(aa) @ bb + np.vstack((self.Kstep_y, np.zeros((1, 2)))) @ ey_k

        goal_step_y = float(np.asarray(u_kp1[0]).squeeze())
        goal_coeff_y = self.Estep_y @ curr_coeff_y + np.linalg.inv(self.Mstep_y) @ u_kp1
        return goal_coeff_y, goal_step_y

    def compute_profile(self, t, coeff, l_step, tk):
        d1 = coeff[0]
        d2 = coeff[1]
        dt = t - tk
        x = d1 * np.exp(l_step * dt) + d2 * np.exp(-l_step * dt)
        v = l_step * (d1 * np.exp(l_step * dt) - d2 * np.exp(-l_step * dt))
        a = l_step * l_step * x
        return x, v, a

    def update_coeff_for_current_step(self, coeff_x, coeff_y, tk, step_x, step_y):
        self.current_coeff_x = coeff_x
        self.current_coeff_y = coeff_y
        self.current_tk = tk
        self.current_step_x = step_x
        self.current_step_y = step_y
        return 1

    def get_des_CoM_state(self, t):
        x_sc_des, x_sc_dot_des, _ = self.compute_profile(
            t, self.current_coeff_x, self.Lstep, self.current_tk
        )
        y_sc_des, y_sc_dot_des, _ = self.compute_profile(
            t, self.current_coeff_y, self.Lstep, self.current_tk
        )
        return x_sc_des, x_sc_dot_des, y_sc_des, y_sc_dot_des

    def set_ankle_PD_gain(self):
        self.kp_x = ANKLE_GAIN["kp_x"]
        self.kd_x = ANKLE_GAIN["kd_x"]
        self.kp_y = ANKLE_GAIN["kp_y"]
        self.kd_y = ANKLE_GAIN["kd_y"]
        self.grav_x = ANKLE_GAIN["grav_x"]
        self.grav_y = ANKLE_GAIN["grav_y"]
        return 0

    def get_ankle_torque(self, t, x_sc, vx_sc, y_sc, vy_sc):
        x_d, vx_d, y_d, vy_d = self.get_des_CoM_state(t)

        tau_y_gravity = -self.mass * self.g * x_sc * self.grav_x
        tau_y_pd = self.mass * self.z_sc_d * (self.kp_x * (x_d - x_sc) + self.kd_x * (vx_d - vx_sc))
        tau_y = tau_y_gravity + tau_y_pd

        tau_x_gravity = self.mass * self.g * y_sc * self.grav_y
        tau_x_pd = -self.mass * self.z_sc_d * (self.kp_y * (y_d - y_sc) + self.kd_y * (vy_d - vy_sc))
        tau_x = tau_x_gravity + tau_x_pd

        return float(np.asarray(tau_x).squeeze()), float(np.asarray(tau_y).squeeze())

    def get_step_length(self):
        return self.current_step_x, self.current_step_y
