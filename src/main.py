import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Гравитационная постоянная в астрономических единицах
G = 2.95912208286e-4
# Массы объектов (Солнце, Земля, Луна)
m0 = 1.0
m1 = 3.00348959632E-6
m2 = m1 * 1.23000383E-2

# Начальные координаты объектов (Солнце, Земля, Луна)
q0 = np.array([
    [0, 0, 0],
    [-0.1667743823220, 0.9690675883429, -0.0000342671456],
    [-0.1694619061456, 0.9692330175719, -0.0000266725711]
])

# Начальные скорости объектов (Солнце, Земля, Луна)
v0 = np.array([
    [0, 0, 0],
    [-0.0172346557280, -0.0029762680930, -0.0000004154391],
    [-0.0172817331582, -0.0035325102831, 0.0000491191454]
])

# Начальные импульсы объектов
p0 = np.array([
    v0[0] * m0,
    v0[1] * m1,
    v0[2] * m2
])

def vecf(v):
    return v / norm(v) ** 3

def fun_v(q):
    f = np.zeros((3, 3))
    f[0] = -G * m0 * m1 * vecf(q[0] - q[1]) - G * m0 * m2 * vecf(q[0] - q[2])
    f[1] = -G * m1 * m0 * vecf(q[1] - q[0]) - G * m1 * m2 * vecf(q[1] - q[2])
    f[2] = -G * m2 * m0 * vecf(q[2] - q[0]) - G * m2 * m1 * vecf(q[2] - q[1])
    return f

def fun_u(p):
    return np.array([p[0] / m0, p[1] / m1, p[2] / m2])

def dormand_prince(t_span, h0, p, q, tol=1e-6):
    vq = [q.copy()]
    vp = [p.copy()]
    distances = []

    t = 0
    h = h0
    while t < t_span:
        k1_q = h * fun_u(p)
        k1_p = h * fun_v(q)

        k2_q = h * fun_u(p + k1_p * 0.2)
        k2_p = h * fun_v(q + k1_q * 0.2)

        k3_q = h * fun_u(p + k1_p * (3 / 40) + k2_p * (9 / 40))
        k3_p = h * fun_v(q + k1_q * (3 / 40) + k2_q * (9 / 40))

        k4_q = h * fun_u(p + k1_p * (44 / 45) - k2_p * (56 / 15) + k3_p * (32 / 9))
        k4_p = h * fun_v(q + k1_q * (44 / 45) - k2_q * (56 / 15) + k3_q * (32 / 9))

        k5_q = h * fun_u(p + k1_p * (19372 / 6561) - k2_p * (25360 / 2187) + k3_p * (64448 / 6561) - k4_p * (212 / 729))
        k5_p = h * fun_v(q + k1_q * (19372 / 6561) - k2_q * (25360 / 2187) + k3_q * (64448 / 6561) - k4_q * (212 / 729))

        k6_q = h * fun_u(p + k1_p * (9017 / 3168) - k2_p * (355 / 33) + k3_p * (46732 / 5247) + k4_p * (49 / 176) - k5_p * (5103 / 18656))
        k6_p = h * fun_v(q + k1_q * (9017 / 3168) - k2_q * (355 / 33) + k3_q * (46732 / 5247) + k4_q * (49 / 176) - k5_q * (5103 / 18656))

        k7_q = h * fun_u(p + k1_p * (35 / 384) + k3_p * (500 / 1113) + k4_p * (125 / 192) - k5_p * (2187 / 6784) + k6_p * (11 / 84))
        k7_p = h * fun_v(q + k1_q * (35 / 384) + k3_q * (500 / 1113) + k4_q * (125 / 192) - k5_q * (2187 / 6784) + k6_q * (11 / 84))

        q_next = q + (35 / 384 * k1_q + 500 / 1113 * k3_q + 125 / 192 * k4_q - 2187 / 6784 * k5_q + 11 / 84 * k6_q)
        p_next = p + (35 / 384 * k1_p + 500 / 1113 * k3_p + 125 / 192 * k4_p - 2187 / 6784 * k5_p + 11 / 84 * k6_p)

        q_err = (q_next - (q + 5179 / 57600 * k1_q + 7571 / 16695 * k3_q + 393 / 640 * k4_q - 92097 / 339200 * k5_q + 187 / 2100 * k6_q + 1 / 40 * k7_q))
        p_err = (p_next - (p + 5179 / 57600 * k1_p + 7571 / 16695 * k3_p + 393 / 640 * k4_p - 92097 / 339200 * k5_p + 187 / 2100 * k6_p + 1 / 40 * k7_p))
        error = np.sqrt(np.sum(q_err ** 2) + np.sum(p_err ** 2))

        if error > tol:
            h = h * 0.9 * (tol / error) ** 0.2
            continue

        q = q_next
        p = p_next
        t += h

        if error < tol / 10:
            h = min(h * 2, h0)

        vq.append(q.copy())
        vp.append(p.copy())
        distances.append(norm(q[2] - q[0]))

    return np.array(vp), np.array(vq), np.array(distances)

# Вычисляет остаточные ошибки для восстановления значений (невязки)
def compute_residual(vp_obs, vq_obs, true_vp, true_vq, masses_obs, true_masses):
    res_p = true_vp - vp_obs
    res_q = true_vq - vq_obs
    res_masses = true_masses - masses_obs
    return np.concatenate((res_p.flatten(), res_q.flatten(), res_masses))

# Вычисляет Якобиан функции f в точке x с использованием численных производных
def compute_jacobian(f, x, epsilon=1e-6):
    n = len(x)
    J = np.zeros((len(f(x)), n))
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon
        J[:, i] = (f(x_perturbed) - f(x)) / epsilon
    return J

# Выполняет LU-разложение матрицы A. Возвращает матрицы L и U
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Верхняя треугольная матрица (U)
        for k in range(i, n):
            U[i, k] = A[i, k] - np.dot(L[i, :i], U[:i, k])

        # Нижняя треугольная матрица (L)
        for k in range(i, n):
            if i == k:
                L[i, i] = 1
            else:
                L[k, i] = (A[k, i] - np.dot(L[k, :i], U[:i, i])) / U[i, i]
    return L, U

# Решает систему уравнений Ax = b с использованием LU-разложения. Возвращает решение x
def solve_lu(L, U, b):
    n = L.shape[0]

    # Прямая подстановка: решаем Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Обратная подстановка: решаем Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Транспонирует матрицу J
def transpose(J):
    rows, cols = J.shape
    JT = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            JT[j, i] = J[i, j]
    return JT

# Добавляет регуляризацию к матрице JTJ для улучшения численной стабильности
def regularization(JTJ, epsilon=1e-6):
    n = JTJ.shape[0]
    return JTJ + epsilon * np.eye(n)

# Решает систему уравнений Ax = b с использованием LU-разложения
def solve(A, b):
    L, U = lu_decomposition(A)
    return solve_lu(L, U, b)

# Вычисляет евклидово расстояние (норму) для вектора v
def norm(v):
    return sum(x ** 2 for x in v) ** 0.5

# Метод Гаусса-Ньютона для восстановления истинных значений параметров
def gauss_newton(true_vp, true_vq, true_masses, vp_obs, vq_obs, masses_obs, max_iterations=15, tolerance=1e-4):
    x = np.concatenate((vp_obs.flatten(), vq_obs.flatten(), masses_obs))
    for iteration in range(max_iterations):
        def residual_function(x):
            vp_obs_current = x[:true_vp.size].reshape(true_vp.shape)
            vq_obs_current = x[true_vp.size:true_vp.size + true_vq.size].reshape(true_vq.shape)
            masses_current = x[true_vp.size + true_vq.size:]
            return compute_residual(vp_obs_current, vq_obs_current, true_vp, true_vq, masses_current,
                                    true_masses)

        J = compute_jacobian(residual_function, x)
        residual = residual_function(x)
        error = norm(residual)
        print(f"Итерация {iteration + 1}, Ошибка: {error}")

        epsilon = 1e-6
        JT = transpose(J)
        JTJ = np.dot(JT, J)
        JTJ_reg = regularization(JTJ, epsilon)
        JTr = np.dot(JT, residual)
        L, U = lu_decomposition(JTJ_reg)
        delta_x = -solve_lu(L, U, JTr.flatten())

        x += delta_x

        if error < tolerance:
            print(f"Сходимость достигнута на {iteration + 1} итерации.")
            break

    vp_restored = x[:true_vp.size].reshape(true_vp.shape)
    vq_restored = x[true_vp.size:true_vp.size + true_vq.size].reshape(true_vq.shape)
    masses_restored = x[true_vp.size + true_vq.size:]

    true_error_vp = norm(vp_restored - true_vp)
    true_error_vq = norm(vq_restored - true_vq)
    true_error_mass = norm(masses_restored - true_masses)
    print(f'Итоговая ошибка относительно истинных значений: для vp: {true_error_vp}, для vq: {true_error_vq}, для mass: {true_error_mass}')

    return vp_restored, vq_restored, masses_restored

def init():
    earth_line.set_data([], [])
    earth_line.set_3d_properties([])
    moon_line.set_data([], [])
    moon_line.set_3d_properties([])
    earth_point.set_data([], [])
    earth_point.set_3d_properties([])
    moon_point.set_data([], [])
    moon_point.set_3d_properties([])
    return earth_line, moon_line, earth_point, moon_point

def update(num):
    earth_line.set_data(vq_restored[:num, 1, 0], vq_restored[:num, 1, 1])
    earth_line.set_3d_properties(vq_restored[:num, 1, 2])

    moon_line.set_data(vq_restored[:num, 2, 0], vq_restored[:num, 2, 1])
    moon_line.set_3d_properties(vq_restored[:num, 2, 2])

    earth_point.set_data([vq_restored[num, 1, 0]], [vq_restored[num, 1, 1]])
    earth_point.set_3d_properties([vq_restored[num, 1, 2]])

    moon_point.set_data([vq_restored[num, 2, 0]], [vq_restored[num, 2, 1]])
    moon_point.set_3d_properties([vq_restored[num, 2, 2]])

    return earth_line, moon_line, earth_point, moon_point


def plot_3d_vp_vq(vq_restored):
    fig = plt.figure(figsize=(12, 6))
    ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    ax2.plot(vq_restored[:, 0, 0], vq_restored[:, 0, 1], vq_restored[:, 0, 2], label='vq[0] (Sun)', linestyle='-', color='orange')
    ax2.plot(vq_restored[:, 1, 0], vq_restored[:, 1, 1], vq_restored[:, 1, 2], label='vq[1] (Earth)', linestyle='-', color='blue')
    ax2.plot(vq_restored[:, 2, 0], vq_restored[:, 2, 1], vq_restored[:, 2, 2], label='vq[2] (Moon)', linestyle='-', color='green')
    ax2.set_title('Восстановленные значения')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    time_span = int(365)
    h = 1
    true_vp, true_vq, _ = dormand_prince(time_span, h, p0, q0)

    noise_level = 1
    m0_obs = m0 + np.random.normal(0, noise_level)
    m1_obs = m1 + np.random.normal(0, noise_level)
    m2_obs = m2 + np.random.normal(0, noise_level)
    masses_obs = np.array([m0_obs, m1_obs, m2_obs])

    print('Массы с наложенное ошибкой: ',m0_obs, m1_obs, m2_obs,'\n')

    vp_obs = true_vp + np.random.normal(0, noise_level, true_vp.shape)
    vq_obs = true_vq + np.random.normal(0, noise_level, true_vq.shape)

    masses_true = np.array([m0, m1, m2])
    vp_restored, vq_restored, masses_restored = gauss_newton(true_vp, true_vq, masses_true, vp_obs, vq_obs, masses_obs)

    plot_3d_vp_vq(vq_restored)

    print("\nВосстановленные массы:", masses_restored)
    print("Истинные массы:", masses_true)
    print("\nВосстановленные динамические параметры:")
    print(f"Импульсы vp: {vp_restored}")
    print(f"Координаты vq: {vq_restored}")

    # Строим график по восстановленным параметрам
    vq_restored[:, 1, :] -= vq_restored[:, 0, :]
    vq_restored[:, 2, :] -= vq_restored[:, 0, :]
    vq_restored[:, 2, :] = vq_restored[:, 1, :] + 100 * (vq_restored[:, 2, :] - vq_restored[:, 1, :])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)
    ax.plot([0], [0], [0], 'yo', markersize=10, label="Солнце")
    earth_line, = ax.plot([], [], [], 'b', label="Земля")
    moon_line, = ax.plot([], [], [], 'r--', label="Луна")
    earth_point, = ax.plot([], [], [], 'bo')
    moon_point, = ax.plot([], [], [], 'ro')

    k = max(1, int(0.1 / h))
    ani = FuncAnimation(fig, update, frames=range(0, len(vq_restored), k), init_func=init, interval=10, blit=True)

    plt.legend()
    plt.show()