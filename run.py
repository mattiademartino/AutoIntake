import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from stl import mesh


from SMARTA import get_trasmission
from create_faces import generate_structured_points, get_inlet, get_output

alpha=0.01
dx=0.001
treeshold=10
modulo=1000
L_intake= 80.0


a= float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
b= float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
c= float(sys.argv[3]) if len(sys.argv) > 3 else 0.0


def plot_performance():
    # Legge il CSV
    df = pd.read_csv("result.csv")

    # Prende la quarta colonna
    col4 = pd.to_numeric(df.iloc[:, 3], errors="coerce")

    # Rimuove eventuali righe non numeriche
    col4 = col4.dropna().reset_index(drop=True)

    # Valore iniziale
    first = col4.iloc[0]

    # Calcolo variazione %
    pct = (col4 - first) / first * 100

    # Iterazioni: 1, 2, 3...
    iterations = range(1, len(pct) + 1)

    # Grafico
    plt.figure(figsize=(8, 4))
    plt.plot(iterations, pct, marker='o')
    plt.title("Andamento performance")
    plt.xlabel("Iterations")
    plt.ylabel("Variazione % rispetto al primo valore")
    plt.grid(True)
    plt.show()

def calculate_L():
    m = mesh.Mesh.from_file("modello.stl")

    points = m.points.reshape(-1, 3)

    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    lunghezza_z = z_max - z_min
    return lunghezza_z

def evaluate_transmission(a, b, c, L):

    present , w = extract_from_csv(a, b, c)

    if present:
        return w

    generate_structured_points(a, b, c, L, verbose=False)

    return get_trasmission(verbose=False)

def archive_all(a,b,c,d,  filename="result.csv"):

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)


    # Aggiungo la nuova riga
    writer.writerow([a, b, c, d])


def extract_from_csv(a, b, c, filename="result.csv"):

    """Cerca una riga con i primi 3 valori uguali. Se trovata, stampa la 4ª colonna."""
    if not os.path.isfile(filename):
        print("Il file result.csv non esiste.")
        return

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # salta l'intestazione se presente

        for row in reader:
            if len(row) < 4:
                continue  # ignora righe malformate

            if row[0] == str(a) and row[1] == str(b) and row[2] == str(c):
                print(row[3])
                return True, float(row[3])

def main():
    get_inlet(verbose=False)
    # get_output richiede points e L come parametri, viene chiamata da generate_structured_points
    iteration = 0
    while True:
        print(f"Iterazione con a={a}, b={b}, c={c}, numero di iterazione={iteration}")
        L= L_intake - calculate_L()

        W_0 = evaluate_transmission(a, b, c, L)
        W_x = evaluate_transmission(a + dx, b, c, L)
        W_y = evaluate_transmission(a, b + dx, c, L)
        W_z = evaluate_transmission(a, b, c + dx, L)

        gradient_a = (W_x - W_0) / dx
        gradient_b = (W_y - W_0) / dx
        gradient_c = (W_z - W_0) / dx
        gradient = np.array([gradient_a, gradient_b, gradient_c])

        archive_all()

        modulo = np.linalg.norm(gradient)
        print(f"Modulo del gradiente: {modulo}")

        if iteration != 0:
            plt.close()
        plot_performance()

        if modulo < treeshold:
            break
        a += alpha * gradient_a
        b += alpha * gradient_b
        c += alpha * gradient_c
        iteration += 1

    








