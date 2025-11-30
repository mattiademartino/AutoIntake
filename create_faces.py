import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys


def generate_structured_points(a, d, e, L):
    """Genera tutti i punti organizzati in 8 griglie strutturate."""
    R = 20
    l = 94
    dz = 80 - L + 0.01

    points = []
    def delta(z):
        return (z / L) * R * (1 - 1 / np.sqrt(2))

    def rho(a, d, e, z):
        b = (R - a * L ** 2 - d * np.sqrt(L) - e * np.exp(-L) - l / np.sqrt(2) + e) / L
        r = a * z ** 2 + b * z + d * np.sqrt(z) + e * np.exp(-z) + l / np.sqrt(2) - e
        return max(r, 1e-6)

    def disp(a, d, e, z, x):
        delt = delta(z)
        r = rho(a, d, e, z)
        limit = 1e-6
        if delt / r < limit:
            return delt * (1 - 2 * (x / r) ** 2)
        else:
            raggio = (2 * delt ** 2 + r ** 2) / (4 * delt)
            radice = raggio ** 2 - x ** 2
            if radice < 0:
                return 0.0
            return np.sqrt(radice) - (raggio - delt)

    def f(a, d, e, x, z):
        return disp(a, d, e, z, x) + rho(a, d, e, z) / np.sqrt(2)

    # Parametri griglia strutturata
    N_z = 50    # Sezioni lungo Z
    N_x = 20    # Punti per sezione X per lato
    
    z_vals = np.linspace(0, L, N_z)
    
    
    for z_idx, z_val in enumerate(z_vals):

        rho_val = rho(a, d, e, z_val)
        
        x_range = rho_val / np.sqrt(2) 
        x_vals = np.linspace(-x_range, x_range, 2*N_x+1) #va da -x_range a +x_range, passando per 0

        for x_idx, x_val in enumerate(x_vals):

            f_val = f(a, d, e, x_val, z_val)
        
            point = {"x": x_val, "f": f_val, "z": z_val, "x_idx": x_idx, "z_idx": z_idx}
            points.append(point)

    return points


def visualize_grid(points):
    """
    Visualizza un insieme di punti 3D.
    
    `points` deve essere una lista/array di vettori,
    dove le prime tre posizioni rappresentano (x, y, z).

    Esempio punto: [x, y, z, ...]
    """

    xs = [point["x"] for point in points]
    ys = [point["z"] for point in points]
    zs = [point["f"] for point in points]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Visualizzazione punti 3D")

    plt.show()


def rotate_points(points, angle_degrees):
    """
    Ruota i punti attorno all'asse Z di un angolo specificato.
    angle_degrees: angolo in gradi (0, 90, 180, 270)
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotated_points = []
    for point in points:
        x = point["x"]
        f = point["f"]
        
        # Rotazione 2D nel piano x-f
        x_new = x * cos_a - f * sin_a
        f_new = x * sin_a + f * cos_a
        
        rotated_point = {
            "x": x_new,
            "f": f_new,
            "z": point["z"],
            "x_idx": point["x_idx"],
            "z_idx": point["z_idx"]
        }
        rotated_points.append(rotated_point)
    
    return rotated_points


def create_stl_from_points(points, filename="face.stl"):
    """
    Crea un file STL da una griglia strutturata di punti.
    Ogni punto ha attributi: x, f, z, x_idx, z_idx
    
    Per ogni punto (x_idx, z_idx), crea due triangoli:
    - Triangolo 1: (x_idx, z_idx), (x_idx, z_idx+1), (x_idx+1, z_idx+1)
    - Triangolo 2: (x_idx, z_idx), (x_idx+1, z_idx), (x_idx+1, z_idx+1)
    """
    
    # Organizza i punti in una mappa per accesso rapido tramite indici
    point_map = {}
    for point in points:
        x_idx = point["x_idx"]
        z_idx = point["z_idx"]
        point_map[(x_idx, z_idx)] = (point["x"], point["f"], point["z"])
    
    triangles = []
    
    # Trova i valori massimi degli indici
    max_x_idx = max(p["x_idx"] for p in points)
    max_z_idx = max(p["z_idx"] for p in points)
    
    # Genera triangoli
    for x_idx in range(max_x_idx):
        for z_idx in range(max_z_idx):
            # Verifica che tutti i punti necessari esistano
            p00 = point_map.get((x_idx, z_idx))
            p01 = point_map.get((x_idx, z_idx + 1))
            p11 = point_map.get((x_idx + 1, z_idx + 1))
            p10 = point_map.get((x_idx + 1, z_idx))
            
            # Primo triangolo: (x_idx, z_idx), (x_idx, z_idx+1), (x_idx+1, z_idx+1)
            if p00 and p01 and p11:
                triangles.append((p00, p01, p11))
            
            # Secondo triangolo: (x_idx, z_idx), (x_idx+1, z_idx), (x_idx+1, z_idx+1)
            if p00 and p10 and p11:
                triangles.append((p00, p10, p11))
    
    # Scrivi il file STL
    with open(filename, 'w') as f:
        f.write("solid generated_mesh\n")
        
        for tri in triangles:
            v1, v2, v3 = tri
            
            # Calcola la normale del triangolo usando il prodotto vettoriale
            edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]])
            edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]])
            normal = np.cross(edge1, edge2)
            
            # Normalizza il vettore normale
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            else:
                normal = np.array([0, 0, 1])
            
            # Scrivi il triangolo in formato STL ASCII
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid generated_mesh\n")
    
    print(f"File STL creato: {filename}")
    print(f"Numero di triangoli: {len(triangles)}")
    return triangles

# Parametri di esempio
a = 0.0
b = 0.0
c = 0.0
L = 80.0

points = generate_structured_points(a , b, c, L)

# Crea la cartella mesh se non esiste
import os
os.makedirs("mesh", exist_ok=True)

# Crea 4 superfici ruotate di 0°, 90°, 180° e 270°
angles = [0, 90, 180, 270]
all_points = []

for angle in angles:
    rotated_points = rotate_points(points, angle)
    filename = f"mesh/face_{angle}deg.stl"
    create_stl_from_points(rotated_points, filename)
    all_points.extend(rotated_points)

# Visualizza tutti i punti insieme (opzionale)
visualize_grid(all_points)

