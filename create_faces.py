import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os
import trimesh
from collections import defaultdict
from stl import mesh
from scipy.spatial import ConvexHull, Delaunay
import warnings

def generate_structured_points(a, d, e, L, verbose=False):
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
    N_z = 10    # Sezioni lungo Z
    N_x = 5    # Punti per sezione X 
    
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


def get_outlet(L, R = 20, filename="mesh/outlet.stl", target_edge_size=5.0, verbose=False):
    """
    Crea una mesh circolare (disco) centrata in (0,0,L) e raggio R sul piano z=L.
    La mesh è composta da triangoli isosceli rettangoli ottenuti dividendo quadrati regolari.
    
    Args:
        L: coordinata z del piano del cerchio
        R: raggio del cerchio
        filename: percorso del file STL di output (ASCII)
        target_edge_size: lunghezza target dell'ipotenusa dei triangoli isosceli rettangoli
        verbose: se True stampa informazioni di debug
    
    Restituisce:
        triangles: lista di triangoli, ciascuno come tuple di 3 vertici (x,y,z)
    """
    if R <= 0:
        raise ValueError("R deve essere > 0")
    if target_edge_size <= 0:
        raise ValueError("target_edge_size deve essere > 0")
    
    # cateto 's' tale che ipotenusa = target_edge_size -> s * sqrt(2) = target_edge_size
    s = target_edge_size / np.sqrt(2.0)
    if s <= 0:
        raise ValueError("Intervallo di griglia non valido (s <= 0)")

    # creare griglia di quadrati che copre [-R, R] x [-R, R]
    xs = np.arange(-R, R + s/2, s)  # +s/2 per includere bordo
    ys = np.arange(-R, R + s/2, s)

    triangles = []

    # per ogni cella quadrata generiamo due triangoli isosceli rettangoli
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[j], ys[j+1]

            # quattro vertici del quadrato (in piano z = L)
            v00 = (x0, y0, L)
            v10 = (x1, y0, L)
            v01 = (x0, y1, L)
            v11 = (x1, y1, L)

            # due triangoli (diagonale v00-v11)
            tri1 = (v00, v10, v11)  # (x0,y0),(x1,y0),(x1,y1)
            tri2 = (v00, v11, v01)  # (x0,y0),(x1,y1),(x0,y1)

            # includi il triangolo se il suo centroide è all'interno del raggio R
            def centroid_inside(tri):
                cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
                cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
                return (cx*cx + cy*cy) <= (R + 1e-12)**2

            if centroid_inside(tri1):
                triangles.append(tri1)
            if centroid_inside(tri2):
                triangles.append(tri2)

    if verbose:
        print(f"Generati {len(triangles)} triangoli isosceli rettangoli (centro in z={L}, raggio={R}).")

    # raccogli vertici unici in una mappa per scrivere file STL
    vertices_map = {}
    def key_of(v):
        return (round(v[0], 6), round(v[1], 6), round(v[2], 6))
    for tri in triangles:
        for v in tri:
            k = key_of(v)
            if k not in vertices_map:
                vertices_map[k] = v

    # funzione per calcolare normale
    def triangle_normal(a, b, c):
        a = np.array(a); b = np.array(b); c = np.array(c)
        e1 = b - a
        e2 = c - a
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm == 0:
            return (0.0, 0.0, 1.0)
        n = n / norm
        return (float(n[0]), float(n[1]), float(n[2]))

    # scrivi STL ASCII
    with open(filename, "w") as f:
        f.write("solid circular_cap\n")
        for tri in triangles:
            v1, v2, v3 = tri
            nx, ny, nz = triangle_normal(v1, v2, v3)
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid circular_cap\n")

    if verbose:
        print(f"File STL scritto: {filename}")

    return triangles

def create_stl_from_points(points, filename="face.stl", verbose=False):
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
    
    if verbose:
        print(f"File STL creato: {filename}")
        print(f"Numero di triangoli: {len(triangles)}")
    return triangles

def get_inlet(input_stl_path: str = "Honeycombs/HC1.STL",
              z_offset: float = 0.1,
              output_dir: str = "mesh",
              n: int = 10,
              verbose: bool = False):
    """
    Estrae i bounds da un file STL e crea un rettangolo inlet sotto il z_min.
    La mesh dell'inlet viene suddivisa in n x n quadrilateri (2 triangoli ciascuno).
    
    Parameters
    ----------
    input_stl_path : str
        Percorso al file STL di input
    z_offset : float
        Distanza sotto z_min a cui posizionare l'inlet
    output_dir : str
        Cartella dove salvare l'inlet STL
    n : int
        Numero di suddivisioni per ciascun lato
    
    Returns
    -------
    (str, float)
        Percorso al file STL dell'inlet creato e valore di z_max
    """

    # Carica il file STL
    stl_mesh = mesh.Mesh.from_file(input_stl_path)

    # Estrai i bounds
    x_min = stl_mesh.x.min()
    x_max = stl_mesh.x.max()
    y_min = stl_mesh.y.min()
    y_max = stl_mesh.y.max()
    z_min = stl_mesh.z.min()
    z_max = stl_mesh.z.max()   # <-- AGGIUNTO

    if verbose:
        print(f"Bounds estratti dal file STL:")
        print(f"  x: [{x_min:.4f}, {x_max:.4f}]")
        print(f"  y: [{y_min:.4f}, {y_max:.4f}]")
        print(f"  z_min: {z_min:.4f}")
        print(f"  z_max: {z_max:.4f}")   # <-- AGGIUNTO

    # Posizione del piano inlet
    z_inlet = z_min - z_offset
    if verbose:
        print(f"  z_inlet: {z_inlet:.4f}")

    # Suddivisione in n intervalli
    xs = np.linspace(x_min, x_max, n + 1)
    ys = np.linspace(y_min, y_max, n + 1)

    # Numero di triangoli
    total_tris = n * n * 2
    inlet_mesh = mesh.Mesh(np.zeros(total_tris, dtype=mesh.Mesh.dtype))

    tri_index = 0
    for i in range(n):
        for j in range(n):
            v0 = [xs[i],     ys[j],     z_inlet]
            v1 = [xs[i+1],   ys[j],     z_inlet]
            v2 = [xs[i],     ys[j+1],   z_inlet]
            v3 = [xs[i+1],   ys[j+1],   z_inlet]

            inlet_mesh.vectors[tri_index] = np.array([v0, v1, v2])
            tri_index += 1

            inlet_mesh.vectors[tri_index] = np.array([v1, v3, v2])
            tri_index += 1

    # Crea cartella output
    os.makedirs(output_dir, exist_ok=True)

    # Nome file output
    input_name = os.path.splitext(os.path.basename(input_stl_path))[0]
    output_path = os.path.join(output_dir, f"{input_name}_inlet.stl")

    # Salva
    inlet_mesh.save(output_path)
    if verbose:
        print(f"\nInlet salvato in: {output_path}")

    # Ritorna anche z_max
    return output_path

if __name__ == "__main__":
    # Parametri di esempio
    a = 0.0
    b = 0.0
    c = 0.0
    L = 80.0

    points = generate_structured_points(a , b, c, L)

    os.makedirs("mesh", exist_ok=True)

    angles = [0, 90, 180, 270]
    all_points = []

    for angle in angles:
        rotated_points = rotate_points(points, angle)
        filename = f"mesh/face{angle//90}.stl"
        create_stl_from_points(rotated_points, filename)
        all_points.extend(rotated_points)


    visualize_grid(all_points)

    #get_inlet("Honeycombs/HC.STL")
    get_inlet()

