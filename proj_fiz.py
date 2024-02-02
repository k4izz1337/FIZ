import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PRĘDKOŚĆ ŚWIATŁA wynosi 299792458 m/s
speed_of_light = 3e8
# EPSILON MASZYNOWE
eps = np.finfo(float).eps


class Particle:
    def __init__(self, pos, vel, mass, charge):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.charge = charge
        self.q_over_m = charge / mass


class Cyclotron:
    spacing = 0.01
    spacing_x0 = -spacing / 2

    # NAPIĘCIE
    voltage = 50000.0
    # POLE ELEKTRYCZNE
    E = [voltage / spacing, 0.0, 0.0]
    # POLE MAGNETYCZNE
    B = [0.0, 0.0, 1.5]
    __b = np.linalg.norm(B)

    # tablica z czasami skoków
    __T_j = []
    # flaga zliczania skoków
    __started_jumping = False
    # prędkość, którą chcemy uzyskać jako wynik
    max_velocity = 0.045 * speed_of_light
    # promień D
    max_R = None
    # maksymalna współrzędna y
    max_y = None
    # czas trzymania cząstki w cyklotronie
    __holding_time = 0.0
    # jeśli prawda, to promień D i maksymalna prędkość będą automatycznie obliczane
    __auto_max_velocity = True
    # jeśli prawda, to znak napięcia będzie zależał od parzystości liczby skoków

    __auto_freq = True
    period = None
    final_velocity = None

    __num_points_of_last_circle = 400

    particle = None
    source_pos = None
    expected_period = None
    delta_t = None

    def set_spacing(self, spacing, x0):
        self.spacing = spacing
        self.spacing_x0 = x0
        self.E = [self.voltage / spacing, 0.0, 0.0]
        if self.expected_period is not None:
            self.delta_t = self.expected_period / (self.__num_points_of_last_circle * math.sqrt(0.05 / spacing))

    def set_voltage(self, vol):
        self.voltage = vol
        self.E = [vol / self.spacing, 0.0, 0.0]

    def set_b(self, b):
        self.__b = math.fabs(b)
        self.B = [0.0, 0.0, b]
        if self.particle is not None:
            self.set_particle(self.particle)

    def set_particle(self, particle):
        self.particle = particle
        self.source_pos = particle.pos
        if self.__b > eps:
            self.expected_period = np.fabs(2.0 * math.pi / (self.__b * self.particle.q_over_m))
            self.delta_t = self.expected_period / (self.__num_points_of_last_circle * math.sqrt(0.05 / self.spacing))
        else:
            self.expected_period = math.inf
            self.delta_t = 1e-10
        self.set_max_velocity(self.max_velocity)

    def set_max_velocity(self, vel):
        assert self.particle is not None, "Particle not set. Use set_particle first"
        self.__auto_max_velocity = True
        self.max_velocity = vel
        if self.__b > eps:
            self.max_R = vel / (self.particle.q_over_m * self.__b)
        else:
            self.max_R = math.inf

    def set_max_R(self, R):
        assert self.particle is not None, "Particle not set. Use set_particle first"
        self.__auto_max_velocity = True
        self.max_R = R
        if self.__b > eps:
            self.max_velocity = self.particle.q_over_m * self.__b * R
        else:
            self.max_velocity = math.inf

    def set_max_y(self, y):
        self.__auto_max_velocity = False
        self.max_y = y

    def set_period(self, period):
        self.__auto_freq = False
        self.period = period

    def enable_auto_freq(self):
        self.__auto_freq = True

    def reset(self):
        self.__started_jumping = False
        self.__T_j = []
        self.__holding_time = 0.0

    def get_result_period(self):
        if len(self.__T_j) == 0:
            return math.inf

        return 2.0 * (self.__T_j[-1] - self.__T_j[0]) / (len(self.__T_j) - 1)

    def is_inside_spacing(self, position):
        return self.spacing_x0 < position[0] < self.spacing_x0 + self.spacing

    def e_acceleration(self, q_over_m, position, time):
        if self.__auto_freq:
            sgn = 1 if len(self.__T_j) % 2 == 0 else -1
        else:
            sgn = 1 if ((time + self.period / 4) // (self.period / 2)) % 2 == 0 else -1
        return np.array(self.E) * (sgn * q_over_m)

    def m_acceleration(self, q_over_m, position, velocity, time):
        return np.cross(velocity, self.B) * q_over_m

# zwraca przyspieszenie spowodowane polem elektromagnetycznym (z siły Lorentza)
    def acceleration(self, q_over_m, position, velocity, time):
        # zatrzymaj się po osiągnięciu maksymalnej prędkości lub maksymalnej wartości y
        if self.__auto_max_velocity:
            if math.fabs(-velocity[0]) >= self.max_velocity and position[1] > self.source_pos[1]:
                return np.zeros(3)
        elif position[1] > self.max_y:
            return np.zeros(3)

        if not self.is_inside_spacing(position):
            if self.__started_jumping:
                self.__started_jumping = False
                self.__T_j.append(time)

            return self.m_acceleration(q_over_m, position, velocity, time)

        self.__started_jumping = True

        return self.e_acceleration(q_over_m, position, time) + self.m_acceleration(q_over_m, position, velocity, time)

    # Metoda Runge-Kutta
    def rk4(self, max_time, delta_t):
        # warunki początkowe
        p = np.array(self.particle.pos)
        v = np.array(self.particle.vel)
        t = 0.0

        P = [p]
        V = [np.linalg.norm(v)]

        for _ in range(0, int(max_time // delta_t)):
            t += delta_t

            p1 = p
            v1 = v
            a1 = delta_t * self.acceleration(self.particle.q_over_m, p1, v1, t)
            v1 = delta_t * v1

            p2 = p + (v1 * 0.5)
            v2 = v + (a1 * 0.5)
            a2 = delta_t * self.acceleration(self.particle.q_over_m, p2, v2, t)
            v2 *= delta_t

            p3 = p + (v2 * 0.5)
            v3 = v + (a2 * 0.5)
            a3 = delta_t * self.acceleration(self.particle.q_over_m, p3, v3, t)
            v3 *= delta_t

            p4 = p + v3
            v4 = v + a3
            a4 = delta_t * self.acceleration(self.particle.q_over_m, p4, v4, t)
            v4 *= delta_t

            dv = a1 + 2.0 * (a2 + a3) + a4
            v = v + dv / 6.0

            dp = v1 + 2.0 * (v2 + v3) + v4
            p = p + dp / 6.0

            # jeśli przyspieszenie jest zakończone, śledź cząstkę w kierunku końcowej prędkości
            if np.allclose(dv, [0., 0., 0.]):
                self.__holding_time = t
                vn = np.linalg.norm(v)
                P = np.concatenate((P, np.linspace(p, p + v * (2 * np.linalg.norm(p) / vn), 100)), axis=0)
                V = np.concatenate((V, np.full(100, vn)), axis=0)
                return P, V

            P.append(p)
            V.append(np.linalg.norm(v))

        self.__holding_time = max_time
        return P, V




    def animate_trajectory(self, max_time, num_frames=200):
        assert self.particle is not None, "Particle not set. Use set_particle first"

        self.reset()

        P, V = self.rk4(max_time, self.delta_t)

        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        def update(frame):
            ax[0].clear()
            ax[1].clear()

            ax[0].scatter(self.source_pos[0], self.source_pos[1], color='blue')

            xc = []
            yc = []
            x = []
            y = []
            for p in P[:int(frame * len(P) / num_frames)]:
                if not self.is_inside_spacing(p):
                    if len(xc):
                        ax[0].plot(xc, yc, color='red', linewidth=0.95)
                        xc = []
                        yc = []
                    x.append(p[0])
                    y.append(p[1])
                else:
                    if len(xc):
                        ax[0].plot(x, y, color='green', linewidth=0.95)
                        x = []
                        y = []
                    xc.append(p[0])
                    yc.append(p[1])

            if len(xc):
                ax[0].plot(xc, yc, color='red', linewidth=0.95)
            if len(x):
                ax[0].plot(x, y, color='green', linewidth=0.95)

            ax[0].axis('equal')
            ax[0].set_title("Trajectory of the particle - Cyclotron")
            ax[0].set_xlabel("Dimension-X (m)")
            ax[0].set_ylabel("Dimension-Y (m)")

            t = np.linspace(0, len(V[:int(frame * len(V) / num_frames)]) * self.delta_t, len(V[:int(frame * len(V) / num_frames)]))
            ax[1].plot(t, V[:int(frame * len(V) / num_frames)])
            ax[1].set_title("Speed of the particle as a function of time")
            ax[1].set_xlabel("Time (sec)")
            ax[1].set_ylabel("Speed (m/s)")

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, repeat=False)
        plt.show()




def main():
    max_time = 1e-5
    d = 0.005
    voltage = 50000.0
    b = 1.5
    max_velocity = 0.05 * speed_of_light
    proton = Particle([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.67E-27, +1.60E-19)

    cyclotron = Cyclotron()
    cyclotron.set_particle(proton)
    cyclotron.set_voltage(voltage)
    cyclotron.set_b(b)
    cyclotron.set_max_velocity(max_velocity)
    cyclotron.set_spacing(d, -d / 2)

    cyclotron.animate_trajectory(max_time, num_frames=200)

if __name__ == '__main__':
    main()