import os
import numpy as np
from stl import mesh

def transform_stl(input_path: str):
    # Costruisci il percorso di output nella stessa directory
    directory = os.path.dirname(input_path)
    output_path = os.path.join(directory, "HC1.STL")

    print(f"Input  file : {input_path}")
    print(f"Output file : {output_path}")

    # Carica la mesh
    m = mesh.Mesh.from_file(input_path)

    # Estrarre tutti i vertici (num_facce * 3 vertici)
    verts = m.vectors.reshape(-1, 3)

    translation = np.array([0.0, 0.0, 150.0])
    verts = verts + translation
    """
    # Matrice rotazione 90° attorno a Y
    R = np.array([
        [0.0, 0.0,  1.0],
        [ 0.0, 1.0,  0.0],
        [ 1.0, 0.0, 0.0]
    ], dtype=float)

    # Applica rotazione
    verts = verts @ R.T

    # Applica traslazione (0, 0, -80)
    translation = np.array([0.0, 0.0, -80.0])
    verts = verts + translation
    """
    # Rimetti i vertici nella mesh
    m.vectors = verts.reshape(-1, 3, 3)

    # Aggiorna normali
    try:
        m.update_normals()
    except:
        pass

    # Salva il nuovo file
    m.save(output_path)
    print("Trasformazione completata.")


# -------------------------------
#   ESECUZIONE
# -------------------------------
transform_stl("Honeycombs/HC1.STL")
