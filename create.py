import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# === ESPORTAZIONE SUPERFICIE IN .STL (ASCII) ===
import trimesh
from scipy.spatial import Delaunay

def points(a, d, e):
    R = 0.02
    l = 0.094
    L = 0.050

    def delta(z):
        return (z / L) * R * (1 - 1 / np.sqrt(2))

    def rho(a, d, e, z):

        # Calcolo del coefficiente b per soddisfare il vincolo rho(L) = R
        b = (R - a * L ** 2 - d * np.sqrt(L) - e * np.exp(-L) - l / np.sqrt(2) + e) / L

        # Definizione di rho(z)
        r = a * z ** 2 + b * z + d * np.sqrt(z) + e * np.exp(-z) + l / np.sqrt(2) - e

        return r

    def disp(a, d, e, z, x):
        delt = delta(z)
        r = rho(a, d, e, z)
        limit=1e-6
        if delt / r < limit:
            return delt * (1 - 2 * (x / r) ** 2)
        else:
            raggio = (2 * delt ** 2 + r ** 2) / (4 * delt)
            radice = raggio ** 2 - x ** 2
            if radice < 0:
                return 5
            return  np.sqrt(radice) -(raggio - delt)

    def f(a, d, e, x, z):
        return disp(a, d, e, z, x) + rho(a, d, e, z) / np.sqrt(2)

    # --- Raccolta per 3D
    N = 100
    z_vals = np.linspace(0.005, L, N)
    all_points_3d = []

    for z_val in z_vals:
        try:
            rho_val = rho(a, d, e, z_val)
            x_vals = np.linspace(-rho_val / np.sqrt(2), rho_val / np.sqrt(2), 200)
            for x_val in x_vals:
                try:
                    f_val = f(a, d, e, x_val, z_val)
                    all_points_3d.append([x_val, f_val, z_val])
                    all_points_3d.append([x_val, -f_val, z_val])
                    all_points_3d.append([f_val, x_val, z_val])
                    all_points_3d.append([-f_val, x_val, z_val])
                except (ZeroDivisionError, ValueError, RuntimeWarning):
                    continue
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            continue

    return np.array(all_points_3d), f, rho, L
