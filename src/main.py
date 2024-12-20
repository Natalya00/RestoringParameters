import numpy as np

G = 2.95912208286e-4

m0 = 1.0
m1 = 3.00348959632E-6
m2 = m1 * 1.23000383E-2

q0 = np.array([
    [0, 0, 0],
    [-0.1667743823220, 0.9690675883429, -0.0000342671456],
    [-0.1694619061456, 0.9692330175719, -0.0000266725711]
])

v0 = np.array([
    [0, 0, 0],
    [-0.0172346557280, -0.0029762680930, -0.0000004154391],
    [-0.0172817331582, -0.0035325102831, 0.0000491191454]
])

level_noisy = 0.1
q_noisy = np.abs(np.random.normal(q0, level_noisy, q0.shape))
v_noisy = np.abs(np.random.normal(v0, level_noisy, v0.shape))
m0_noisy = np.abs(np.random.normal(m0, level_noisy))
m1_noisy = np.abs(np.random.normal(m1, level_noisy))
m2_noisy = np.abs(np.random.normal(m2, level_noisy))

p0 = np.array([
    v0[0] * m0,
    v0[1] * m1,
    v0[2] * m2
])

p0_noisy = np.array([
    v_noisy[0] * m0_noisy,
    v_noisy[1] * m1_noisy,
    v_noisy[2] * m2_noisy
])

print("Зашумленные параметры:")
print("m0:", m0_noisy)
print("m1:", m1_noisy)
print("m2:", m2_noisy)
print("q_n:", q_noisy)
print("v_n:", v_noisy)

def vecf(v):
    return v / np.linalg.norm(v) ** 3


def fun_v(q, m0, m1, m2):
    f = np.zeros((3, 3))
    f[0] = -G * m0 * m1 * vecf(q[0] - q[1]) - G * m0 * m2 * vecf(q[0] - q[2])
    f[1] = -G * m1 * m0 * vecf(q[1] - q[0]) - G * m1 * m2 * vecf(q[1] - q[2])
    f[2] = -G * m2 * m0 * vecf(q[2] - q[0]) - G * m2 * m1 * vecf(q[2] - q[1])
    return f


def fun_u(p, m0, m1, m2):
    return np.array([p[0] / m0, p[1] / m1, p[2] / m2])


def dormand_prince(t_span, h0, p, q, m0, m1, m2, tol=1e-6):
    vq = [q.copy()]
    vp = [p.copy()]
    distances = []

    t = 0
    h = h0
    while t < t_span:
        k1_q = h * fun_u(p, m0, m1, m2)
        k1_p = h * fun_v(q, m0, m1, m2)

        k2_q = h * fun_u(p + k1_p * 0.2, m0, m1, m2)
        k2_p = h * fun_v(q + k1_q * 0.2, m0, m1, m2)

        k3_q = h * fun_u(p + k1_p * (3 / 40) + k2_p * (9 / 40), m0, m1, m2)
        k3_p = h * fun_v(q + k1_q * (3 / 40) + k2_q * (9 / 40), m0, m1, m2)

        k4_q = h * fun_u(p + k1_p * (44 / 45) - k2_p * (56 / 15) + k3_p * (32 / 9), m0, m1, m2)
        k4_p = h * fun_v(q + k1_q * (44 / 45) - k2_q * (56 / 15) + k3_q * (32 / 9), m0, m1, m2)

        k5_q = h * fun_u(p + k1_p * (19372 / 6561) - k2_p * (25360 / 2187) + k3_p * (64448 / 6561) - k4_p * (212 / 729), m0, m1, m2)
        k5_p = h * fun_v(q + k1_q * (19372 / 6561) - k2_q * (25360 / 2187) + k3_q * (64448 / 6561) - k4_q * (212 / 729), m0, m1, m2)

        k6_q = h * fun_u(p + k1_p * (9017 / 3168) - k2_p * (355 / 33) + k3_p * (46732 / 5247) + k4_p * (49 / 176) - k5_p * (5103 / 18656), m0, m1, m2)
        k6_p = h * fun_v(q + k1_q * (9017 / 3168) - k2_q * (355 / 33) + k3_q * (46732 / 5247) + k4_q * (49 / 176) - k5_q * (5103 / 18656), m0, m1, m2)

        q_next = q + (35 / 384 * k1_q + 500 / 1113 * k3_q + 125 / 192 * k4_q - 2187 / 6784 * k5_q + 11 / 84 * k6_q)
        p_next = p + (35 / 384 * k1_p + 500 / 1113 * k3_p + 125 / 192 * k4_p - 2187 / 6784 * k5_p + 11 / 84 * k6_p)

        q = q_next
        p = p_next
        t += h

        vq.append(q.copy())
        vp.append(p.copy())
        distances.append(np.linalg.norm(q[2] - q[0]))

    return np.array(vp)[-1], np.array(vq)[-1], np.array(distances)[-1]

# Функция для вычисления матрицы Якобиана
# Вычисляет частные производные по параметрам массы (m0_n, m1_n, m2_n), координатам q_n и скоростям v_n
def compute_jacobian(q_n, v_n, m0_n, m1_n, m2_n, true_distances, epsilon=1e-3):
    # Вычисление текущих импульсов на основе скоростей и масс
    p_n = np.array([v_n[0] * m0_n, v_n[1] * m1_n, v_n[2] * m2_n])
    jacobian = np.zeros((1, 3 + q_n.size + v_n.size))
    t_span = 365

    # Вычисляем частные производные по массам m0, m1, m2
    for j in range(3):
        perturbed_masses = np.array([m0_n, m1_n, m2_n])
        perturbed_masses[j] += epsilon # Прибавляем малое значение epsilon к одной из масс
        perturbed_p_n = np.array([v_n[0] * perturbed_masses[0], v_n[1] * perturbed_masses[1], v_n[2] * perturbed_masses[2]]) # Заново вычисляем импульсы, так как изменили массы
        _, _, perturbed_distances = dormand_prince(t_span, 1, perturbed_p_n, q_n, *perturbed_masses) # Пересчитываем расстояние с текущими параметрами
        jacobian[:, j] = (perturbed_distances - true_distances) / epsilon

    # Вычисляем частные производные по координатам q
    q_flat = q_n.ravel()
    for j in range(q_flat.size):
        perturbed_q = q_flat.copy()
        perturbed_q[j] += epsilon # Прибавляем малое значение epsilon к координатам
        perturbed_q = perturbed_q.reshape(q_n.shape)
        _, _, perturbed_distances = dormand_prince(t_span, 1, p_n, perturbed_q, m0_n, m1_n, m2_n)
        jacobian[:, 3 + j] = (perturbed_distances - true_distances) / epsilon

    # Вычисляем частные производные по скоростям v
    v_flat = v_n.ravel()
    for j in range(v_flat.size):
        perturbed_v = v_flat.copy()
        perturbed_v[j] += epsilon # Прибавляем малое значение epsilon к скоростям
        perturbed_v = perturbed_v.reshape(v_n.shape)
        perturbed_p_n = np.array([perturbed_v[0] * m0_n, perturbed_v[1] * m1_n, perturbed_v[2] * m2_n])
        _, _, perturbed_distances = dormand_prince(t_span, 1, perturbed_p_n, q_n, m0_n, m1_n, m2_n)
        jacobian[:, 3 + q_flat.size + j] = (perturbed_distances - true_distances) / epsilon

    return jacobian

# Транспонирование матрицы
def transpose(matrix):
    rows, cols = matrix.shape
    transposed = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = matrix[i, j]
    return transposed

# Умножение матриц A и B
def matrix_multiply(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    result = np.zeros((rows_A, cols_B))
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
    return result

# LU-разложение матрицы A на нижнюю треугольную (L) и верхнюю треугольную (U) матрицы
def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i, i] = 1
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    return L, U

# Решение системы линейных уравнений LUX = b методом подстановки
def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i)) # Прямой ход (решение L*y = b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i] # Обратный ход (решение U*x = y)
    return x

def gauss_newton(iterations, noisy_params, true_distances, h, t_span):
    m0_n, m1_n, m2_n, q_n, v_n = noisy_params # Значения с шумом, которые необходимо восстановить

    for i in range(iterations):
        p_n = [v_n[0] * m0_n, v_n[1] * m1_n, v_n[2] * m2_n] # Вычисляем текущие импульсы
        _, _, distances = dormand_prince(t_span, h, p_n, q_n, m0_n, m1_n, m2_n) # Вычисляем расстояния по текущим параметрам

        residuals = np.array(true_distances - distances).reshape((1, 1)) # Разница между наблюдаемыми и предсказанными расстояниями
        residual_norm = sum(r[0] ** 2 for r in residuals) ** 0.5 # Норма вектора невязки

        if residual_norm < 1e-6: # Условие остановки
            break

        print(residual_norm)

        jacobian = compute_jacobian(q_n, v_n, m0_n, m1_n, m2_n, true_distances) # Вычисление Якобиана

        jacobian_T = transpose(jacobian) # Транспонирование Якобиана
        J_T_J = matrix_multiply(jacobian_T, jacobian) # J^T * J

        for i in range(len(J_T_J)):
            J_T_J[i][i] += 1e-1 # Добавление регуляризации

        J_T_residuals = matrix_multiply(jacobian_T, residuals) # J^T * r

        L, U = lu_decomposition(J_T_J) # LU-разложение матрицы J^T * J
        delta = solve_lu(L, U, [-r[0] for r in J_T_residuals]) # Решение системы линейных уравнений

        # Обновление параметров
        m0_n += delta[0]
        m1_n += delta[1]
        m2_n += delta[2]

        for i in range(len(q_n)):
            q_n[i] += delta[3 + i]

        for i in range(len(v_n)):
            v_n[i] += delta[3 + len(q_n) + i]

    return m0_n, m1_n, m2_n, q_n, v_n

time_span = int(365)
h = 1

# Истинные значения
true_vp, true_vq, true_distances = dormand_prince(time_span, h, p0, q0, m0, m1, m2)

# Восстановление параметров
noisy_params = (m0_noisy, m1_noisy, m2_noisy, q_noisy, v_noisy)

restored_params = gauss_newton(20, noisy_params, true_distances, h, time_span)

# Вывод результатов
print("Восстановленные параметры:")
print("m0:", restored_params[0])
print("m1:", restored_params[1])
print("m2:", restored_params[2])
print("q_n:", restored_params[3])
print("v_n:", restored_params[4])
