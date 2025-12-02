# Copyright (C) 2020  Pietro Parodi
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#############################################################################
# SMARTA Simulation script                                                  #
#############################################################################

import sys
import os
import signal

# Aggiungi la cartella SMARTA_functions al path per importare rarfunc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SMARTA_functions'))

from rarfunc import *

# Handler per permettere l'interruzione con Ctrl+C in MPI
def signal_handler(sig, frame):
    print('\n\nInterruzione richiesta (Ctrl+C). Terminazione in corso...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def get_trasmission(verbose=True):
    mpicomm = Comm()
    universe = Universe()

    rprint(mpicomm, 'Loading mesh...')


    #############################################################################
    # Load mesh                                                                 #
    #############################################################################
    folder = './mesh/'
    universe.add(mpicomm, folder+'face0.stl', 0, 'wall') 
    universe.add(mpicomm, folder+'face1.stl', 0, 'wall') 
    universe.add(mpicomm, folder+'face2.stl', 0, 'wall') 
    universe.add(mpicomm, folder+'face3.stl', 0, 'wall') 
    universe.add(mpicomm, folder+'HC_inlet.stl',  1, 'inlet') 
    universe.add(mpicomm, folder+'outlet.stl', 2, 'output')


    folder = './Honeycombs/'
    universe.add(mpicomm, folder+'HC1.STL',   0,'wall')



    #############################################################################
    # Initialize universe                                                       #
    #############################################################################



    universe.init(mpicomm)

    if verbose:
        rprint(mpicomm, 'Loaded a total of {N} triangles.'.format(N=universe.N))

    mpicomm.comm.Barrier()

    #############################################################################
    universe.prop(0, T=300.0)
    universe.prop(1, S=2.0, n=1e16, T=800, m=4e-26, uhat=[0.0,0.0,1.0])


    mpicomm.comm.Barrier()


    #############################################################################
    # Compute view factor matrix or load from file                              #
    #############################################################################
    LOADFMATRIX = False
    filename = '/nobackup/st/parodi/Fmatrix.npy'

    if LOADFMATRIX == False:
        if verbose:
            rprint(mpicomm, 'Computing view factors matrix...')
        tic(mpicomm)
        F = view_factors(mpicomm, universe)
        if verbose:
            toc(mpicomm, 'View factor matrix computation')
    else:
        if mpicomm.rank == 0:
            F = np.load(filename)
        else:
            F = None

    if verbose:
        rprint(mpicomm, 'Maximum value in F matrix is:  {val}'.format(val=np.max(F)))
        rprint(mpicomm, 'Minimum value in F matrix is:  {val}'.format(val=np.min(F)))

    #############################################################################
    # Saving of view factor matrix to file for later use                        #
    # Caution: could be multiple GB!                                            #
    #############################################################################
    SAVEFMATRIX = False
    filename = '/nobackup/st/parodi/Fmatrix.npy'

    if SAVEFMATRIX == True:
        if mpicomm.rank == 0:
            np.save(filename, F)


    #############################################################################
    # Show a map of the view factor matrix                                      #
    # Caution: may take some time                                               #
    #############################################################################
    SHOWFIMAGE = False

    if SHOWFIMAGE == True:
        if mpicomm.rank == 0:
            plt.imshow(np.log10(np.clip(F, 1e-4, np.max(F))), interpolation='none', cmap='binary')
            plt.colorbar()
            plt.show()

    #############################################################################
    # Calculation of matrices F1 and F2                                         #
    #############################################################################

    if verbose:
        rprint(mpicomm, 'Computing F1 matrix...')

    F1 = compute_F1(mpicomm, universe, F)

    if verbose:
        rprint(mpicomm, 'Completed.')

    F2 = F

    #############################################################################
    # Calculation of matrix M                                                   #
    #############################################################################

    if verbose:
        rprint(mpicomm, 'Computing M matrix...')
    tic(mpicomm)

    M = compute_M(mpicomm, universe)

    if verbose:
        toc(mpicomm, 'M matrix computation')
        rprint(mpicomm, 'Maximum value in M matrix is:  {val}'.format(val=np.max(M)))

    if verbose:
        rprint(mpicomm, 'Computing E matrix...')

    E = compute_E(mpicomm, universe)

    #############################################################################
    # Solution of the linear system with np.linalg.solve (alternatives shown)   #
    #############################################################################

    if mpicomm.rank == 0:
        tic(mpicomm)
        b = np.dot(M * F2, E)
        if verbose:
            print('Solving the linear matrix equation...')
        try:
            B = np.linalg.solve(F1, b)
            # B, lstsqresiduals, lstsqrank, lstsqs = np.linalg.lstsq(F1, b)
            # B, rnorm = nnls(F1, b)
        except np.linalg.LinAlgError as e:
            print(str(e))
            raise
        if verbose:
            toc(mpicomm, 'Solving the linear matrix equation')
    else:
        B = None
    mpicomm.comm.Barrier()


    if mpicomm.rank == 0:
        bcheck = np.dot(F1, B)
        if verbose:
            print('Completed. Checking if solution is exact: {res}'.format(res = np.allclose(bcheck, b)))



    #############################################################################
    # Computation of total flux through a surface group                         #
    #                                                                           #
    # tot_flux(Comm, Universe, B, ID)                                           #
    #   Calculates the total flux.                                              #
    #       - Comm is an object of the Comm class (MPI communicator info)       #
    #       - Universe is an object of the Universe class (mesh and gas info)   #
    #       - ID is the int identifier of the group of surfaces to calculate    #
    #         the total flux on                                                 #
    #       - B is the surface flux vector                                      #
    #############################################################################
        
    if mpicomm.rank == 0:
        fout = tot_flux(universe, B, 0)


    return fout
