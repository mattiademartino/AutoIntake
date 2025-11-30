import numpy as np
import trimesh
import sys

# ============================================================================
# CONFIGURAZIONE GEOMETRICA
# ============================================================================

class GridConfig:
    """Parametri geometrici della griglia."""
    TOTAL_SIZE = 100.0
    WALL_THICKNESS = 0.5
    FRAME_THICKNESS = 3.0
    TARGET_EDGE_SIZE = 5.0  # Dimensione target degli edge in mm per triangoli isosceli
    
    def __init__(self, n, L):
        self.n = n
        self.L = L
        self.internal_size = self.TOTAL_SIZE - 2 * self.FRAME_THICKNESS
        self.hole_spacing = self.internal_size / n
        self.hole_size = self.hole_spacing - self.WALL_THICKNESS
        
        # Calcola suddivisioni ottimali per triangoli isosceli
        self.n_x = max(2, int(L / self.TARGET_EDGE_SIZE))  # Suddivisioni assiali
        self.n_yz = max(2, int(self.TOTAL_SIZE / self.TARGET_EDGE_SIZE))  # Suddivisioni trasversali
        self.actual_edge_x = L / self.n_x
        self.actual_edge_yz = self.TOTAL_SIZE / self.n_yz
        
    def is_valid(self):
        """Verifica se la configurazione è geometricamente possibile."""
        return self.hole_size > 0 and 0 < self.n <= 50 and 0 < self.L <= 1000

# ============================================================================
# OPERAZIONI MESH BASE
# ============================================================================

def create_base_box(config):
    """Crea il parallelepipedo base con discretizzazione ottimale."""
    # Centro in (0, 0, z) con una faccia in z=0 e l'altra in z=-L
    x_center = 0
    y_center = 0
    z_center = -config.L / 2
    
    # Crea griglia di vertici strutturata
    x_vals = np.linspace(-config.TOTAL_SIZE / 2, config.TOTAL_SIZE / 2, config.n_yz + 1)
    y_vals = np.linspace(-config.TOTAL_SIZE / 2, config.TOTAL_SIZE / 2, config.n_yz + 1)
    z_vals = np.linspace(-config.L, 0, config.n_x + 1)
    
    vertices = []
    faces = []
    
    # Genera vertici e facce per ogni faccia del box
    # Faccia superiore (z=0)
    offset = 0
    for i in range(config.n_yz + 1):
        for j in range(config.n_yz + 1):
            vertices.append([x_vals[i], y_vals[j], 0])
    for i in range(config.n_yz):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    
    # Faccia inferiore (z=-L)
    offset = len(vertices)
    for i in range(config.n_yz + 1):
        for j in range(config.n_yz + 1):
            vertices.append([x_vals[i], y_vals[j], -config.L])
    for i in range(config.n_yz):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v2, v1], [v1, v2, v3]])
    
    # Faccia laterale X- (x=-TOTAL_SIZE/2)
    offset = len(vertices)
    for i in range(config.n_x + 1):
        for j in range(config.n_yz + 1):
            vertices.append([-config.TOTAL_SIZE / 2, y_vals[j], z_vals[i]])
    for i in range(config.n_x):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v2, v1], [v1, v2, v3]])
    
    # Faccia laterale X+ (x=TOTAL_SIZE/2)
    offset = len(vertices)
    for i in range(config.n_x + 1):
        for j in range(config.n_yz + 1):
            vertices.append([config.TOTAL_SIZE / 2, y_vals[j], z_vals[i]])
    for i in range(config.n_x):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    
    # Faccia laterale Y- (y=-TOTAL_SIZE/2)
    offset = len(vertices)
    for i in range(config.n_x + 1):
        for j in range(config.n_yz + 1):
            vertices.append([x_vals[j], -config.TOTAL_SIZE / 2, z_vals[i]])
    for i in range(config.n_x):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v1, v2], [v1, v3, v2]])
    
    # Faccia laterale Y+ (y=TOTAL_SIZE/2)
    offset = len(vertices)
    for i in range(config.n_x + 1):
        for j in range(config.n_yz + 1):
            vertices.append([x_vals[j], config.TOTAL_SIZE / 2, z_vals[i]])
    for i in range(config.n_x):
        for j in range(config.n_yz):
            v0 = offset + i * (config.n_yz + 1) + j
            v1 = v0 + 1
            v2 = v0 + (config.n_yz + 1)
            v3 = v2 + 1
            faces.extend([[v0, v2, v1], [v1, v2, v3]])
    
    box = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    box.merge_vertices()
    return box

def create_hole(config, i, j):
    """Crea un singolo foro alle coordinate (i, j)."""
    z_center = -config.L / 2
    hole_x_center = -config.TOTAL_SIZE / 2 + config.FRAME_THICKNESS + config.hole_spacing * (i + 0.5)
    hole_y_center = -config.TOTAL_SIZE / 2 + config.FRAME_THICKNESS + config.hole_spacing * (j + 0.5)
    
    hole = trimesh.creation.box(extents=[config.hole_size, config.hole_size, config.L + 0.1])
    hole.apply_translation([hole_x_center, hole_y_center, z_center])
    return hole

def clean_mesh(mesh):
    """Pulisce e ottimizza una mesh."""
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    try:
        mesh.fill_holes()
    except:
        pass
    return mesh

# ============================================================================
# GENERAZIONE METODO CSG
# ============================================================================

def generate_csg_grid(config):
    """Genera griglia usando operazioni booleane CSG."""
    print(f"Generando griglia {config.n}×{config.n}, L={config.L}")
    print(f"Discretizzazione: {config.n_x} suddivisioni assiali, {config.n_yz}×{config.n_yz} trasversali")
    print(f"Edge size: X={config.actual_edge_x:.2f}mm, YZ={config.actual_edge_yz:.2f}mm")
    
    result_mesh = create_base_box(config)
    holes_created = 0
    
    for i in range(config.n):
        for j in range(config.n):
            try:
                hole = create_hole(config, i, j)
                new_mesh = result_mesh.difference(hole)
                
                if new_mesh is not None and len(new_mesh.faces) > 0:
                    result_mesh = new_mesh
                    holes_created += 1
            except:
                continue
    
    print(f"Fori creati: {holes_created}/{config.n * config.n}")
    return clean_mesh(result_mesh)

# ============================================================================
# GENERAZIONE METODO MANUALE
# ============================================================================

def create_wall_component(config, x_start, x_end, y_start, y_end):
    """Crea un componente rettangolare della griglia."""
    if x_start >= x_end or y_start >= y_end:
        return None
    
    try:
        width = x_end - x_start
        height = y_end - y_start
        depth = config.L
        
        box = trimesh.creation.box(extents=[width, height, depth])
        
        center_x = (x_start + x_end) / 2
        center_y = (y_start + y_end) / 2
        center_z = -config.L / 2
        
        box.apply_translation([center_x, center_y, center_z])
        return box
    except:
        return None

def generate_manual_grid(config):
    """Costruisce la griglia manualmente componente per componente."""
    print(f"Costruendo griglia {config.n}×{config.n} manualmente")
    
    components = []
    
    # Cornice esterna
    components.append(create_wall_component(config, -config.TOTAL_SIZE/2, config.TOTAL_SIZE/2, -config.TOTAL_SIZE/2, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS))
    components.append(create_wall_component(config, -config.TOTAL_SIZE/2, config.TOTAL_SIZE/2, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS, config.TOTAL_SIZE/2))
    components.append(create_wall_component(config, -config.TOTAL_SIZE/2, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS))
    components.append(create_wall_component(config, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS, config.TOTAL_SIZE/2, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS))
    
    # Pareti orizzontali (lungo Y)
    for i in range(config.n - 1):
        y_center = -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS + (i + 1) * config.hole_spacing
        y_start = y_center - config.WALL_THICKNESS / 2
        y_end = y_center + config.WALL_THICKNESS / 2
        components.append(create_wall_component(config, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS, y_start, y_end))
    
    # Pareti verticali (lungo X)
    for j in range(config.n - 1):
        x_center = -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS + (j + 1) * config.hole_spacing
        x_start = x_center - config.WALL_THICKNESS / 2
        x_end = x_center + config.WALL_THICKNESS / 2
        components.append(create_wall_component(config, x_start, x_end, -config.TOTAL_SIZE/2 + config.FRAME_THICKNESS, config.TOTAL_SIZE/2 - config.FRAME_THICKNESS))
    
    # Filtra componenti None
    components = [c for c in components if c is not None]
    
    if not components:
        return None
    
    # Combina tutti i componenti
    result = components[0]
    for comp in components[1:]:
        try:
            result = trimesh.util.concatenate([result, comp])
        except:
            continue
    
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    result.remove_duplicate_faces()
    
    return result

# ============================================================================
# SALVATAGGIO E VALIDAZIONE
# ============================================================================

def save_mesh(mesh, filename):
    """Salva la mesh in formato STL."""
    try:
        mesh.export(filename, file_type='stl_ascii')
        return True
    except:
        return False

def print_mesh_info(mesh, filename):
    """Stampa informazioni sulla mesh generata."""
    print(f"\nMesh: {filename}")
    print(f"  Vertici: {len(mesh.vertices)}, Facce: {len(mesh.faces)}")
    print(f"  Volume: {mesh.volume:.2f} mm³")
    print(f"  Watertight: {'✓' if mesh.is_watertight else '✗'}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Funzione principale."""
    if len(sys.argv) != 3:
        print("Usage: python create_HC.py n L")
        print("  n: numero di fori per lato (1-50)")
        print("  L: lunghezza dello sweep (0-1000)")
        return
    
    try:
        n = int(sys.argv[1])
        L = float(sys.argv[2])
        config = GridConfig(n, L)
        
        if not config.is_valid():
            print(f"Errore: configurazione non valida (hole_size={config.hole_size:.3f})")
            return
        
        # Crea la cartella mesh se non esiste
        import os
        os.makedirs("mesh", exist_ok=True)
        
        filename = "mesh/HC.stl"
        
        # Tentativo 1: Metodo CSG
        mesh = generate_csg_grid(config)
        
        # Tentativo 2: Metodo manuale (se necessario)
        if mesh is None or not mesh.is_watertight:
            print("\nTentativo con metodo manuale...")
            filename = "mesh/HCg_manual.stl"
            mesh = generate_manual_grid(config)
        
        # Salva e mostra risultati
        if mesh is not None:
            if save_mesh(mesh, filename):
                print_mesh_info(mesh, filename)
                if mesh.is_watertight:
                    print("\n✓ Successo: mesh watertight generata")
                else:
                    print("\n⚠ Attenzione: mesh non watertight")
            else:
                print("\nErrore nel salvataggio del file")
        else:
            print("\nErrore: impossibile generare la mesh")
            
    except ValueError:
        print("Errore: n deve essere intero, L deve essere numero")
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()
