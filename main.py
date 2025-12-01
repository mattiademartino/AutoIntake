#!/usr/bin/env python3
"""
Python version of main.sh optimization script.
Uses gradient descent to optimize parameters a, b, c.
Calls get_transmission() from SMARTA.py instead of running SPARTA via mpiexec.
"""

import sys
import subprocess
import numpy as np
from SMARTA import get_trasmission


def run_command(cmd):
    """Run a shell command and return its output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Command failed: {cmd}")
        print(f"Error: {result.stderr}")
    return result.stdout.strip()


def calculate_L():
    """Calculate L using calculate_L.py script."""
    result = run_command("python3 calculate_L.py")
    return float(result)


def create_geometry(a, b, c, L):
    """Create geometry using create.py."""
    cmd = f"python3 create.py {a} {b} {c} {L}"
    run_command(cmd)


def merge_meshes():
    """Merge HC.stl with other meshes."""
    # Assuming merge.py takes HC.stl as argument
    run_command("python3 merge.py mesh/HC.stl")


def convert_stl_to_surf():
    """Convert merge.stl to merge.surf."""
    run_command("python3 stl2surf.py merge.stl merge.surf")


def read_and_save(a, b, c, L):
    """Save results using read_and_save.py."""
    cmd = f"python3 read_and_save.py {a} {b} {c} {L}"
    run_command(cmd)


def calculate_gradient(a, b, c, L):
    """Calculate gradient using grad.py and return grada, gradb, gradc."""
    cmd = f"python3 grad.py {a} {b} {c} {L}"
    run_command(cmd)
    
    # Read gradients from grad.out
    try:
        with open("grad.out", "r") as f:
            lines = f.readlines()
            grada = float(lines[0].strip())
            gradb = float(lines[1].strip())
            gradc = float(lines[2].strip())
        return grada, gradb, gradc
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Error reading gradients: {e}")
        return 0.0, 0.0, 0.0


def run_simulation_and_save(a, b, c, L):
    """Run complete simulation pipeline for given parameters."""
    print(f"  Running simulation for a={a}, b={b}, c={c}")
    
    # Create geometry
    create_geometry(a, b, c, L)
    
    # Merge meshes
    merge_meshes()
    
    # Convert to surf format (if needed)
    # convert_stl_to_surf()
    
    # Run simulation using SMARTA
    transmission = get_trasmission()
    
    # Save results
    read_and_save(a, b, c, L)
    
    return transmission


def main():
    # Parameters
    alpha = 0.01
    dx = 0.001
    soglia = 10.0
    modulo = 1000.0
    
    # Initialize parameters with defaults or command line arguments
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    c = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    
    print(f"Parametri iniziali: a={a}, b={b}, c={c}")
    
    # Calculate L
    L = calculate_L()
    print(f"L calcolato: {L}")
    
    iteration = 0
    
    # Main optimization loop
    while modulo >= soglia:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iterazione {iteration}")
        print(f"{'='*60}")
        
        # Central point simulation
        print("Simulazione punto centrale:")
        run_simulation_and_save(a, b, c, L)
        
        # Perturbed simulations for numerical derivative
        print("\nSimulazioni perturbate per derivata numerica:")
        
        # a + dx
        print(f"1/3: Perturbazione su a")
        run_simulation_and_save(a + dx, b, c, L)
        
        # b + dx
        print(f"2/3: Perturbazione su b")
        run_simulation_and_save(a, b + dx, c, L)
        
        # c + dx
        print(f"3/3: Perturbazione su c")
        run_simulation_and_save(a, b, c + dx, L)
        
        # Calculate gradient
        print("\nCalcolo gradiente...")
        grada, gradb, gradc = calculate_gradient(a, b, c, L)
        
        # Calculate gradient magnitude
        modulo = np.sqrt(grada**2 + gradb**2 + gradc**2)
        print(f"Modulo gradiente: {modulo:.6f}")
        
        # Check convergence
        if modulo < soglia:
            print(f"\nConvergenza raggiunta! Modulo gradiente ({modulo:.6f}) < soglia ({soglia})")
            break
        
        # Update parameters
        a = a + alpha * grada
        b = b + alpha * gradb
        c = c + alpha * gradc
        
        print(f"\nNuovi parametri: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    
    print(f"\n{'='*60}")
    print("Ottimizzazione completata!")
    print(f"{'='*60}")
    print(f"Parametri finali: a={a:.6f}, b={b:.6f}, c={c:.6f}")
    print(f"Iterazioni totali: {iteration}")


if __name__ == "__main__":
    main()
