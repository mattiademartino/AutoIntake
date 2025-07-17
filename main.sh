#!/bin/bash

alpha=0.01
dx=0.001
soglia=10
modulo=1000

# Inizializza i parametri con valori di default
a=${1:-0.0}
b=${2:-0.0}
c=${3:-0.0}

echo "Parametri iniziali: a=$a, b=$b, c=$c"

# Calcola L 
L=$(python3 calculate_L.py)

while (( $(echo "$soglia <= $modulo" | bc -l) )); do
	# Simulazione punto centrale
	python3 create.py $a $b $c $L
	python3 merge HC.stl
    python3 stl2surf.py merge.stl merge.surf
	mpiexec -np 4 ~/sparta-20Jan2025/src/spa_mpi -in in.sim
	python3 read_and_save.py $a $b $c $L

	# Simulazioni perturbate per derivata numerica
	python3 create.py $(echo "$a + $dx" | bc -l) $b $c $L
	python3 merge HC.stl
    python3 stl2surf.py merge.stl merge.surf
	mpiexec -np 4 ~/sparta-20Jan2025/src/spa_mpi -in in.sim
	python3 read_and_save.py $(echo "$a + $dx" | bc -l) $b $c $L

	python3 create.py $a $(echo "$b + $dx" | bc -l) $c $L
	python3 merge HC.stl
    python3 stl2surf.py merge.stl merge.surf
	mpiexec -np 4 ~/sparta-20Jan2025/src/spa_mpi -in in.sim
	python3 read_and_save.py $a $(echo "$b + $dx" | bc -l) $c $L

	python3 create.py $a $b $(echo "$c + $dx" | bc -l) $L
	python3 merge HC.stl
    python3 stl2surf.py merge.stl merge.surf
	mpiexec -np 4 ~/sparta-20Jan2025/src/spa_mpi -in in.sim
	python3 read_and_save.py $a $b $(echo "$c + $dx" | bc -l) $L

	# Calcolo gradiente (scrive in grad.out: 3 righe con grada, gradb, gradc)
	python3 grad.py $a $b $c $L

	# Leggi i gradienti da file
	read grada < grad.out
	read gradb < <(sed -n 2p grad.out)
	read gradc < <(sed -n 3p grad.out)

	# Calcola modulo del gradiente
	modulo=$(echo "scale=6; sqrt($grada^2 + $gradb^2 + $gradc^2)" | bc -l)
	echo "Modulo gradiente: $modulo"

	# Aggiorna i parametri
	a=$(echo "$a + $alpha * $grada" | bc -l)
	b=$(echo "$b + $alpha * $gradb" | bc -l)
	c=$(echo "$c + $alpha * $gradc" | bc -l)

	echo "Nuovi parametri: a=$a, b=$b, c=$c"
done
