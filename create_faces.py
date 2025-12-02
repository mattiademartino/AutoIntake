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
    N_z = 50    # Sezioni lungo Z
    N_x = 20    # Punti per sezione X 
    
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


def get_output(points, L, filename="/mesh/outlet.stl", target_edge_size=5.0, verbose=False):
    """
    Crea una mesh circolare che chiude l'estremità a z=L.
    I punti a z=L formano un cerchio, questa funzione crea triangoli
    quanto più isosceli rettangoli possibili.
    
    Args:
        points: lista di punti con attributi x, f, z
        L: valore di z dove si trova il cerchio
        filename: nome del file STL da creare
        target_edge_size: dimensione target dei lati dei triangoli
        verbose: se True, stampa messaggi di output
    """
    # Estrai i punti a z=L
    edge_points = [p for p in points if abs(p["z"] - L) < 1e-6]
    
    if len(edge_points) == 0:
        if verbose:
            print(f"Nessun punto trovato a z={L}")
        return []
    
    # Ordina i punti in senso orario/antiorario attorno al centro
    coords = np.array([[p["x"], p["f"]] for p in edge_points])
    
    # Calcola il centro
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    center = np.array([center_x, center_y])
    
    # Calcola angoli rispetto al centro
    angles = np.arctan2(coords[:, 1] - center_y, coords[:, 0] - center_x)
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    
    # Calcola il raggio medio
    radius = np.mean(np.linalg.norm(sorted_coords - center, axis=1))
    
    # Calcola quanti anelli radiali servono per avere triangoli di dimensione target
    n_rings = max(1, int(radius / target_edge_size))
    
    triangles = []
    vertices_map = {}
    
    # Funzione helper per aggiungere vertice
    def add_vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertices_map:
            vertices_map[key] = (x, y, z)
        return key
    
    # Crea anelli concentrici
    n_edge_points = len(sorted_coords)
    
    # Vertice centrale
    center_key = add_vertex(center_x, center_y, L)
    
    # Crea vertici per ogni anello
    for ring in range(1, n_rings + 1):
        ring_radius = radius * (ring / n_rings)
        
        for i in range(n_edge_points):
            angle = angles[sorted_indices[i]]
            x = center_x + ring_radius * np.cos(angle)
            y = center_y + ring_radius * np.sin(angle)
            add_vertex(x, y, L)
    
    # Aggiungi i punti del bordo esterno (quelli originali)
    for coord in sorted_coords:
        add_vertex(coord[0], coord[1], L)
    
    # Crea lista ordinata di vertici per ogni anello
    rings_vertices = []
    
    # Anello 0: solo il centro
    rings_vertices.append([center_key])
    
    # Anelli intermedi
    for ring in range(1, n_rings + 1):
        ring_radius = radius * (ring / n_rings)
        ring_verts = []
        for i in range(n_edge_points):
            angle = angles[sorted_indices[i]]
            x = center_x + ring_radius * np.cos(angle)
            y = center_y + ring_radius * np.sin(angle)
            key = (round(x, 6), round(y, 6), round(L, 6))
            ring_verts.append(key)
        rings_vertices.append(ring_verts)
    
    # Anello esterno (punti originali)
    outer_ring = []
    for coord in sorted_coords:
        key = (round(coord[0], 6), round(coord[1], 6), round(L, 6))
        outer_ring.append(key)
    rings_vertices.append(outer_ring)
    
    # Crea triangoli tra anelli
    for ring_idx in range(len(rings_vertices) - 1):
        inner_ring = rings_vertices[ring_idx]
        outer_ring = rings_vertices[ring_idx + 1]
        
        if len(inner_ring) == 1:
            # Dal centro al primo anello: triangoli semplici
            center_v = inner_ring[0]
            for i in range(len(outer_ring)):
                v1 = outer_ring[i]
                v2 = outer_ring[(i + 1) % len(outer_ring)]
                triangles.append((center_v, v1, v2))
        else:
            # Tra anelli: crea quad e dividili in triangoli
            for i in range(len(inner_ring)):
                v1_inner = inner_ring[i]
                v2_inner = inner_ring[(i + 1) % len(inner_ring)]
                v1_outer = outer_ring[i]
                v2_outer = outer_ring[(i + 1) % len(outer_ring)]
                
                # Due triangoli per quad
                triangles.append((v1_inner, v1_outer, v2_outer))
                triangles.append((v1_inner, v2_outer, v2_inner))
    
    # Scrivi il file STL
    with open(filename, 'w') as f:
        f.write("solid circular_cap\n")
        
        for tri_keys in triangles:
            v1 = vertices_map[tri_keys[0]]
            v2 = vertices_map[tri_keys[1]]
            v3 = vertices_map[tri_keys[2]]
            
            # Calcola la normale del triangolo
            edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]])
            edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]])
            normal = np.cross(edge1, edge2)
            
            # Normalizza
            norm_length = np.linalg.norm(normal)
            if norm_length > 0:
                normal = normal / norm_length
            else:
                normal = np.array([0, 0, 1])
            
            # Scrivi il triangolo
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid circular_cap\n")
    
    if verbose:
        print(f"File STL cap creato: {filename}")
        print(f"Numero di triangoli: {len(triangles)}")
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

