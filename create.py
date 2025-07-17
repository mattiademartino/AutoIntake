import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from scipy.spatial import Delaunay
import sys

def points(a, d, e):
    R = 0.02
    l = 0.094
    L = 0.050

    def delta(z):
        return (z / L) * R * (1 - 1 / np.sqrt(2))

    def rho(a, d, e, z):
        b = (R - a * L ** 2 - d * np.sqrt(L) - e * np.exp(-L) - l / np.sqrt(2) + e) / L
        r = a * z ** 2 + b * z + d * np.sqrt(z) + e * np.exp(-z) + l / np.sqrt(2) - e
        return r

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
                return 5
            return np.sqrt(radice) - (raggio - delt)

    def f(a, d, e, x, z):
        return disp(a, d, e, z, x) + rho(a, d, e, z) / np.sqrt(2)

    # --- Raccolta per 3D
    N = 50
    z_vals = np.linspace(0.005, L, N)
    all_points_3d = []

    for z_idx, z_val in enumerate(z_vals):
        try:
            rho_val = rho(a, d, e, z_val)
            x_vals = np.linspace(-rho_val / np.sqrt(2), rho_val / np.sqrt(2), 30)

            for x_idx, x_val in enumerate(x_vals):
                try:
                    f_val = f(a, d, e, x_val, z_val)
                    all_points_3d.append([x_val,  f_val,  z_val, z_idx, x_idx, 1])
                    all_points_3d.append([x_val, -f_val,  z_val, z_idx, x_idx, 2])
                    all_points_3d.append([f_val,  x_val,  z_val, z_idx, x_idx, 3])
                    all_points_3d.append([-f_val, x_val,  z_val, z_idx, x_idx, 4])

                    #Costruiamo anche un piano parallelo in modo da poter creare qualcosa di 3D e non watertight
                    all_points_3d.append([x_val,  f_val+0.001,  z_val, z_idx, x_idx, 5])
                    all_points_3d.append([x_val, -(f_val+0.001),  z_val, z_idx, x_idx, 6])
                    all_points_3d.append([f_val+0.001,  x_val,  z_val, z_idx, x_idx, 7])
                    all_points_3d.append([-(f_val+0.001), x_val,  z_val, z_idx, x_idx, 8])
                except (ZeroDivisionError, ValueError, RuntimeWarning):
                    continue
        except (ZeroDivisionError, ValueError, RuntimeWarning):
            continue

    return np.array(all_points_3d)

def create_stl_from_points(all_points_3d, filename="output.stl"):
    """
    Create STL file from the points array following the specified instructions:
    - For each block (1-8), create 2D mesh with triangles
    - Connect block pairs (1-5), (2-6), (3-7), (4-8) at extremities
    - Unite the 4 resulting blocks
    """
    
    # Separate points by block
    blocks = {}
    for i in range(1, 9):
        block_points = all_points_3d[all_points_3d[:, 5] == i]
        if len(block_points) > 0:
            blocks[i] = block_points
    
    all_vertices = []
    all_faces = []
    vertex_count = 0
    
    def add_triangle(v1, v2, v3):
        """Add a triangle to the mesh with proper normal orientation"""
        nonlocal vertex_count
        all_vertices.extend([v1, v2, v3])
        face = [vertex_count, vertex_count + 1, vertex_count + 2]
        all_faces.append(face)
        vertex_count += 3
    
    def get_point_by_indices(block_data, z_idx, x_idx):
        """Get point coordinates by z_idx and x_idx"""
        point = block_data[(block_data[:, 3] == z_idx) & (block_data[:, 4] == x_idx)]
        if len(point) > 0:
            return point[0][:3]  # Return x, y, z coordinates
        return None
    
    def calculate_normal(v1, v2, v3):
        """Calculate face normal using cross product"""
        edge1 = np.array(v2) - np.array(v1)
        edge2 = np.array(v3) - np.array(v1)
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            return normal / norm
        return np.array([0, 0, 1])
    
    # Create 2D meshes for each block
    for block_id in range(1, 9):
        if block_id not in blocks:
            continue
            
        block_data = blocks[block_id]
        
        # Get unique z_idx and x_idx values for this block
        z_indices = np.unique(block_data[:, 3]).astype(int)
        x_indices = np.unique(block_data[:, 4]).astype(int)
        
        # Create triangles for the 2D mesh
        for z_idx in z_indices[:-1]:  # Skip last z_idx to avoid out of bounds
            for x_idx in x_indices[:-1]:  # Skip last x_idx to avoid out of bounds
                
                # Get the four corner points of the quad
                p1 = get_point_by_indices(block_data, z_idx, x_idx)
                p2 = get_point_by_indices(block_data, z_idx + 1, x_idx)
                p3 = get_point_by_indices(block_data, z_idx, x_idx + 1)
                p4 = get_point_by_indices(block_data, z_idx + 1, x_idx + 1)
                
                # Only create triangles if all points exist
                if p1 is not None and p2 is not None and p3 is not None and p4 is not None:
                    # First triangle: (z_idx,x_idx), (z_idx+1,x_idx), (z_idx,x_idx+1)
                    add_triangle(p1, p2, p3)
                    
                    # Second triangle: (z_idx+1,x_idx), (z_idx,x_idx+1), (z_idx+1,x_idx+1)
                    add_triangle(p2, p3, p4)
    
    # Connect block pairs at extremities
    block_pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]
    
    for block1_id, block2_id in block_pairs:
        if block1_id not in blocks or block2_id not in blocks:
            continue
            
        block1_data = blocks[block1_id]
        block2_data = blocks[block2_id]
        
        # Get z extremities (z_idx = 0 and z_idx = max)
        z_indices = np.unique(block1_data[:, 3]).astype(int)
        x_indices = np.unique(block1_data[:, 4]).astype(int)
        
        z_min, z_max = z_indices[0], z_indices[-1]
        
        # Connect at z_min extremity
        for x_idx in x_indices[:-1]:
            p1_b1 = get_point_by_indices(block1_data, z_min, x_idx)
            p2_b1 = get_point_by_indices(block1_data, z_min, x_idx + 1)
            p1_b2 = get_point_by_indices(block2_data, z_min, x_idx)
            p2_b2 = get_point_by_indices(block2_data, z_min, x_idx + 1)
            
            if all(p is not None for p in [p1_b1, p2_b1, p1_b2, p2_b2]):
                add_triangle(p1_b1, p1_b2, p2_b1)
                add_triangle(p2_b1, p1_b2, p2_b2)
        
        # Connect at z_max extremity
        for x_idx in x_indices[:-1]:
            p1_b1 = get_point_by_indices(block1_data, z_max, x_idx)
            p2_b1 = get_point_by_indices(block1_data, z_max, x_idx + 1)
            p1_b2 = get_point_by_indices(block2_data, z_max, x_idx)
            p2_b2 = get_point_by_indices(block2_data, z_max, x_idx + 1)
            
            if all(p is not None for p in [p1_b1, p2_b1, p1_b2, p2_b2]):
                add_triangle(p1_b1, p2_b1, p1_b2)
                add_triangle(p2_b1, p2_b2, p1_b2)
    
    # Convert to numpy arrays
    vertices = np.array(all_vertices)
    faces = np.array(all_faces)
    
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Remove duplicate vertices and fix normals
    try:
        # Use updated method for removing duplicate faces
        mesh.update_faces(mesh.unique_faces())
    except:
        # Fallback for older versions
        try:
            mesh.remove_duplicate_faces()
        except:
            pass
    
    mesh.remove_unreferenced_vertices()
    
    # Try to fix normals, but don't fail if networkx is not available
    try:
        mesh.fix_normals()
    except ImportError:
        print("Warning: Cannot fix normals (networkx not installed). STL may have inconsistent face orientations.")
    except Exception as e:
        print(f"Warning: Could not fix normals: {e}")
    
    # Export to STL in ASCII format
    mesh.export(filename, file_type='stl_ascii')
    print(f"STL file saved as: {filename} (ASCII format)")
    print(f"Mesh info: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh

# Example usage
if __name__ == "__main__":
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print("Usage: python3 points.py a d e")
        print("Example: python3 points.py 0.1 0.05 0.02")
        sys.exit(1)
    
    try:
        # Parse command line arguments
        a = float(sys.argv[1])
        d = float(sys.argv[2])
        e = float(sys.argv[3])
    except ValueError:
        print("Error: All parameters must be numeric values")
        print("Usage: python3 points.py a d e")
        print("Example: python3 points.py 0.1 0.05 0.02")
        sys.exit(1)
    
    print(f"Parameters: a={a}, d={d}, e={e}")
    
    print("Generating points...")
    points_3d = points(a, d, e)
    print(f"Generated {len(points_3d)} points")
    
    # Create filename based on parameters
    filename = f"mesh.stl"
    
    print("Creating STL file...")
    mesh = create_stl_from_points(points_3d, filename)
 """   
    # Optional: visualize the mesh
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a sample of points for visualization
        sample_points = points_3d[::100]  # Sample every 100th point
        ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                  c=sample_points[:, 5], cmap='tab10', s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Generated 3D Points (a={a}, d={d}, e={e})')
        plt.show()
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("STL file created successfully without visualization")
        """
