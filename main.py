import numpy as np
import matplotlib.pyplot as plt


class PendulumMotion:
    def __init__(self):
        self.g = 9.81
        self.r = 1.0
        self.m = 1.0
        self.dt = 0.01
        self.T = 10
        self.steps = int(self.T / self.dt)

        self.alpha0 = np.radians(45)
        self.omega0 = 0.0
        self.t = np.linspace(0, self.T, self.steps)

    def compute_energy(self, alpha, omega):
        Ep = self.m * self.g * self.r * (1 - np.cos(alpha))
        Ek = 0.5 * self.m * (self.r * omega) ** 2
        return Ep, Ek, Ep + Ek

    def improved_euler(self):
        alpha = np.zeros(self.steps)
        omega = np.zeros(self.steps)
        Ep = np.zeros(self.steps)
        Ek = np.zeros(self.steps)
        E = np.zeros(self.steps)

        alpha[0] = self.alpha0
        omega[0] = self.omega0
        Ep[0], Ek[0], E[0] = self.compute_energy(alpha[0], omega[0])

        for i in range(1, self.steps):
            a1 = omega[i - 1]
            w1 = (self.g / self.r) * np.sin(alpha[i - 1])

            a2 = omega[i - 1] + self.dt * w1
            w2 = (self.g / self.r) * np.sin(alpha[i - 1] + self.dt * a1)

            alpha[i] = alpha[i - 1] + self.dt * 0.5 * (a1 + a2)
            omega[i] = omega[i - 1] + self.dt * 0.5 * (w1 + w2)

            Ep[i], Ek[i], E[i] = self.compute_energy(alpha[i], omega[i])

        return alpha, omega, Ep, Ek, E

    def rk4(self):
        alpha = np.zeros(self.steps)
        omega = np.zeros(self.steps)
        Ep = np.zeros(self.steps)
        Ek = np.zeros(self.steps)
        E = np.zeros(self.steps)

        alpha[0] = self.alpha0
        omega[0] = self.omega0
        Ep[0], Ek[0], E[0] = self.compute_energy(alpha[0], omega[0])

        for i in range(1, self.steps):
            a = alpha[i - 1]
            w = omega[i - 1]

            k1_a = w
            k1_w = (self.g / self.r) * np.sin(a)

            k2_a = w + 0.5 * self.dt * k1_w
            k2_w = (self.g / self.r) * np.sin(a + 0.5 * self.dt * k1_a)

            k3_a = w + 0.5 * self.dt * k2_w
            k3_w = (self.g / self.r) * np.sin(a + 0.5 * self.dt * k2_a)

            k4_a = w + self.dt * k3_w
            k4_w = (self.g / self.r) * np.sin(a + self.dt * k3_a)

            alpha[i] = a + (self.dt / 6) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
            omega[i] = w + (self.dt / 6) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)

            Ep[i], Ek[i], E[i] = self.compute_energy(alpha[i], omega[i])

        return alpha, omega, Ep, Ek, E

    def plot_results(self, Ep, Ek, E, title):
        plt.figure(figsize=(10, 5))
        plt.plot(self.t, Ep, label='Potential Energy', linestyle="-")
        plt.plot(self.t, Ek, label='Kinetic Energy', linestyle="-")
        plt.plot(self.t, E, label='Total Energy', linestyle="-")
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("\tChoose a simulation")
    print("Euler or rk4 (e / r )")

    method = input(": ")
    sim = PendulumMotion()

    if method == "e":
        alpha, omega, Ep, Ek, E = sim.improved_euler()
        sim.plot_results(Ep, Ek, E, "Improved Euler")

    elif method == "r":
        alpha, omega, Ep, Ek, E = sim.rk4()
        sim.plot_results(Ep, Ek, E, "RK4")

    else:
        print("wrong user input")
