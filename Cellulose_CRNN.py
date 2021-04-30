import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


class CRNN():
    def __init__(self):

        self.Arrhenius = np.array([[222.4, 0, 17.3],
                                   [117.2, 0.15, 14.92],
                                   [218.0, 0.34, 36.75],
                                   [88.7, 0.05, 22.69],
                                   [110.5, 0.03, 14.08],
                                   [187.4, 0.04, 33.64]])
        self.lb = 1e-8
        self.lnO2 = np.log(self.lb)
        self.R = 8.314E-3

    def cal_T(self, t):

        return 700 + 273.15

    def cal_w(self, t, Y):

        T = self.cal_T(t)

        lnY = np.log(Y.clip(self.lb))
        Cellu = lnY[0]
        S2 = lnY[1]
        S3 = lnY[2]

        lnA = self.Arrhenius[:, 2]
        b = self.Arrhenius[:, 1]
        Ea = self.Arrhenius[:, 0]
        lnk = lnA + b*np.log(T) - Ea / self.R / T

        w = np.zeros(6)

        w[0] = 0.2 * Cellu + lnk[0]
        w[1] = 0.4 * Cellu + 0.63 * S3 + lnk[1]
        w[2] = 1.15 * Cellu + 0.38 * S2 + lnk[2]
        w[3] = 1.32 * S3 + 0.45 * self.lnO2 + lnk[3]
        w[4] = 1.91 * S2 + 0.33 * self.lnO2 + lnk[4]
        w[5] = 1.52 * Cellu + 0.19 * self.lnO2 + lnk[5]

        return np.exp(w)

    def __call__(self, t, Y):

        w = self.cal_w(t, Y)

        dy = np.zeros(4)

        dy[0] = -0.2 * w[0] - 0.4 * w[1] - 1.15 * w[2] - 1.52 * w[5]
        dy[1] = 0.41 * w[1] - 0.38 * w[2] + \
            0.56 * w[3] - 1.91 * w[4] + 0.38 * w[5]
        dy[2] = 0.2 * w[0] - 0.61 * w[1] + 0.64 * \
            w[2] - 1.32 * w[3] + 1.27 * w[4] + 0.46 * w[5]
        dy[3] = 0.61 * w[1] + 0.89 * w[2] + \
            0.76 * w[3] + 0.63 * w[4] + 0.68 * w[5]

        return dy


if __name__ == '__main__':

    y0 = [1, 0, 0, 0]

    ode = CRNN()

    sol = solve_ivp(ode,
                    t_span=[0, 10000],
                    y0=y0,
                    # t_eval=[],
                    method='BDF',
                    dense_output=False,
                    vectorized=False,
                    rtol=1e-3,
                    atol=ode.lb)

    # cal rates
    sol_w = np.zeros((6, sol.t.shape[0]))

    for i in range(sol.t.shape[0]):
        sol_w[:, i] = ode.cal_w(sol.t[i], sol.y[:, i])

    fig = plt.figure(figsize=(8, 4))
    varnames = ["Cellu", "S2", "S3", "Vola"]
    for i in range(4):
        plt.semilogx(sol.t + 1e-4, sol.y[i, :], label=varnames[i])
    plt.xlabel('Time [s]')
    plt.ylabel('Y')
    plt.legend()
    fig.tight_layout()
    plt.savefig('species_profile', dpi=200)
    plt.show()

    fig = plt.figure(figsize=(9, 6))
    for i in range(6):
        fig.add_subplot(2, 3, i+1)
        plt.semilogx(sol.t + 1e-4, sol_w[i, :], label="R"+str(i+1))
        plt.xlabel('Time [s]')
        plt.ylabel('w')
        plt.legend()
    fig.tight_layout()
    plt.savefig('rates_profile', dpi=200)
    plt.show()
