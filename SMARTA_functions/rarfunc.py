#############################################################################
# RAREFIED Utility functions script                                         #
#############################################################################

import numpy as np
import scipy as sp
from scipy.optimize import nnls
import math
from stl import mesh
import matplotlib.pyplot as plt
import time

from mpi4py import MPI

import vtk
from vtk.util.numpy_support import numpy_to_vtk

import rarfast

k = 1.38064852e-23

def tic(mpicomm):
    if mpicomm.rank == 0:
        global t
        t = time.time()

def toc(mpicomm, event):
    if mpicomm.rank == 0:
        global t
        elapsed = time.time() - t
        print('{event} took {elapsed:.2f} s.'.format(event=event, elapsed=elapsed))
        tic(mpicomm)

def split_proc(seq, size):
    newseq = [[] for i in range(size)]
    for i in seq:
        newseq[i % size].append(i)
        
    flatseq = [item for sublist in newseq for item in sublist]
    offsets=[]
    for i in range(len(seq)):
        offsets.append(flatseq.index(i))
        
    return newseq, offsets


def flux_thermal(n, T, m):
    global k
    return n*np.sqrt(8*k*T/(np.pi*m))/4

def flux_hyper(n, T, m, S):
    return  flux_thermal(n, T, m) * ( np.exp(-S*S) + np.sqrt(np.pi) * S * (1 + math.erf(S)) )


def view_factors(mpicomm, universe):

    mychunk = universe.chunks[mpicomm.rank]
    local_F = np.zeros((len(mychunk), universe.N))

    rarfast.compute_F(universe.N, mychunk, local_F, universe.centers, universe.normals, universe.v1, universe.v2, universe.v3, universe.areas)
    
    local_F.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    mpicomm.comm.Barrier()
    if mpicomm.rank == 0:
        F = np.empty(sum(sendcounts), dtype=float)
    else:
        F = None

    mpicomm.comm.Gatherv(sendbuf=local_F, recvbuf=(F, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        F=np.reshape(F, (universe.N, universe.N))
        F=F[universe.offsets]
        F=(F+F.T)*universe.areas


    mpicomm.comm.Barrier()
    return F


def compute_F1(mpicomm, universe, F):
    mpicomm.comm.Barrier()
    if mpicomm.rank==0:
        F1=(-F*universe.rhos) + np.identity(universe.N)
    else:
        F1 = None
    mpicomm.comm.Barrier()
    return F1



def compute_M(mpicomm, universe):
    mpicomm.comm.Barrier()
    mychunk = universe.chunks[mpicomm.rank]
    
    local_M = np.zeros((len(mychunk), universe.N))

    rarfast.compute_M(universe.N, mychunk, universe.IDs, universe.types, universe.uhat, universe.S, local_M, universe.centers)

    local_M.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        M = np.empty(sum(sendcounts), dtype=float)
    else:
        M = None

    mpicomm.comm.Gatherv(sendbuf=local_M, recvbuf=(M, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        M=np.reshape(M, (universe.N,universe.N))
        M=M[universe.offsets]
        M=M.T
        
##    if rank == 0:
##        printProgressBar(1, 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    mpicomm.comm.Barrier()
    return M



def compute_E(mpicomm, universe):
    
    mychunk = universe.chunks[mpicomm.rank]
    local_E = np.zeros(len(mychunk))
    for ii, i in enumerate(mychunk):
        if universe.types[i] == 1:
            local_E[ii] = flux_thermal(universe.n[universe.IDs[i]], universe.T[universe.IDs[i]], universe.m)


    sendcounts = [len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        E = np.empty(sum(sendcounts), dtype=float)
    else:
        E = None

    mpicomm.comm.Gatherv(sendbuf=local_E, recvbuf=(E, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        E=E[universe.offsets]
        
##    if rank == 0:
##        printProgressBar(1, 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    mpicomm.comm.Barrier()
    return E




def compute_s(mpicomm, universe):

    mychunk = universe.chunks[mpicomm.rank]

    local_sx = np.zeros((len(mychunk), universe.N))
    local_sy = np.zeros((len(mychunk), universe.N))
    local_sz = np.zeros((len(mychunk), universe.N))

    rarfast.compute_s(universe.N, mychunk, universe.centers, local_sx, local_sy, local_sz)
                

    local_sx.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        sx = np.empty(sum(sendcounts), dtype=float)
    else:
        sx = None

    mpicomm.comm.Gatherv(sendbuf=local_sx, recvbuf=(sx, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        sx=np.reshape(sx, (universe.N,universe.N))
        sx=sx[universe.offsets]
        sx=sx-sx.T

    local_sy.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        sy = np.empty(sum(sendcounts), dtype=float)
    else:
        sy = None

    mpicomm.comm.Gatherv(sendbuf=local_sy, recvbuf=(sy, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        sy=np.reshape(sy, (universe.N,universe.N))
        sy=sy[universe.offsets]
        sy=sy-sy.T

    local_sz.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        sz = np.empty(sum(sendcounts), dtype=float)
    else:
        sz = None

    mpicomm.comm.Gatherv(sendbuf=local_sz, recvbuf=(sz, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        sz=np.reshape(sz, (universe.N,universe.N))
        sz=sz[universe.offsets]
        sz=sz-sz.T

    mpicomm.comm.Barrier()
##    if rank == 0:
##        printProgressBar(1, 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    return sx, sy, sz


def compute_P(mpicomm, universe):
    mychunk = universe.chunks[mpicomm.rank]

    local_P = np.zeros((len(mychunk), universe.N))

    rarfast.compute_P(universe.N, mychunk, universe.IDs, universe.types, universe.uhat, universe.S, local_P, universe.centers)

    local_P.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        P = np.empty(sum(sendcounts), dtype=float)
    else:
        P = None

    mpicomm.comm.Gatherv(sendbuf=local_P, recvbuf=(P, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        P=np.reshape(P, (universe.N,universe.N))
        P=P[universe.offsets]
        
##    if rank == 0:
##        printProgressBar(1, 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    mpicomm.comm.Barrier()
    return P


def compute_pressure(mpicomm, universe, F, B):
    global k
    if mpicomm.rank == 0:
        p1 = np.zeros(universe.N)
        p2 = np.zeros(universe.N)

        for i in range(universe.N):
            if universe.types[i] == 0: #a wall or outlet
                p1[i] = 3/4*np.sqrt(2*np.pi*k*universe.T[universe.IDs[i]]*universe.m)*universe.rhos[i]*B[i]
                p2[i] = 1/2*np.sqrt(2*np.pi*k*universe.T[universe.IDs[i]]*universe.m)*universe.rhos[i]*B[i]
            if universe.types[i] == 1: #an emitting surface
                p1[i] = 3/4*universe.n[universe.IDs[i]]*k*universe.T[universe.IDs[i]]

    mpicomm.comm.Barrier()
    
    P = compute_P(mpicomm, universe)

    sx, sy, sz = compute_s(mpicomm, universe)

    if mpicomm.rank == 0:
        

        nx = [normal[0] for normal in universe.normals]
        ny = [normal[1] for normal in universe.normals]
        nz = [normal[2] for normal in universe.normals]

        px = universe.rhos*np.dot(p1, (np.transpose(F)*P*sx)) - p2*nx
        py = universe.rhos*np.dot(p1, (np.transpose(F)*P*sy)) - p2*ny
        pz = universe.rhos*np.dot(p1, (np.transpose(F)*P*sz)) - p2*nz

        p = np.sqrt(px*px + py*py + pz*pz)

        return px, py, pz, p
    else:
        return None, None, None, None


def compute_Ene(mpicomm, universe):
    mychunk = universe.chunks[mpicomm.rank]

    local_Ene = np.zeros((len(mychunk), universe.N))

    rarfast.compute_Ene(universe.N, mychunk, universe.IDs, universe.types, universe.uhat, universe.S, local_Ene, universe.centers)

    local_Ene.flatten()
    sendcounts = [universe.N*len(eachchunk) for eachchunk in universe.chunks]
    if mpicomm.rank == 0:
        Ene = np.empty(sum(sendcounts), dtype=float)
    else:
        Ene = None

    mpicomm.comm.Gatherv(sendbuf=local_Ene, recvbuf=(Ene, sendcounts), root=0)
    
    if mpicomm.rank == 0:
        Ene=np.reshape(Ene, (universe.N,universe.N))
        Ene=Ene[universe.offsets]
        
##    if rank == 0:
##        printProgressBar(1, 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    mpicomm.comm.Barrier()
    return Ene

def compute_energy(mpicomm, universe, F, B):
    global k
    if mpicomm.rank == 0:
        Ein  = np.zeros(universe.N)
        Eout = np.zeros(universe.N)
        
        for i in range(universe.N):
            
            
            if universe.types[i] == 0: #a wall or outlet
                Ein[i] =  2*k*universe.T[universe.IDs[i]]*universe.rhos[i]*B[i]
                Eout[i] = 2*k*universe.T[universe.IDs[i]]*universe.rhos[i]*B[i]
            if universe.types[i] == 1: #an emitting surface
                Ein[i] = 2*k*universe.T[universe.IDs[i]]*1/4*universe.n[universe.IDs[i]]*np.sqrt(8*k*universe.T[universe.IDs[i]]/(np.pi*universe.m))

    mpicomm.comm.Barrier()
    
    Ene = compute_Ene(mpicomm, universe)

    if mpicomm.rank == 0:

        energy = universe.rhos*np.dot(Ein, np.transpose(F)*Ene) - Eout

        return energy
    else:
        return None

    
def tot_flux(universe, B, ID):
    flux = 0
    for i in range(len(B)):
        if universe.IDs[i] == ID:
            flux += B[i] * universe.areas[i]
    return flux

def tot_force(mpicomm, universe, ID, B, px, py, pz):
    if mpicomm.rank == 0:
        force = np.zeros(3)
        for i in range(len(B)):
            if universe.IDs[i] == ID:
                force += np.array([px[i], py[i], pz[i]]) * universe.areas[i]
        return force
    else:
        return [None, None, None]


def tot_moment(mpicomm, universe, ID, B, px, py, pz, point):
    if mpicomm.rank == 0:
        moment = np.zeros(3)
        for i in range(len(B)):
            if universe.IDs[i] == ID:
                force = np.array([px[i], py[i], pz[i]]) * universe.areas[i]
                arm = universe.centers[i]-point
                moment += np.cross(arm, force)
        return moment
    else:
        return [None, None, None]

def write_vtk(mpicomm, universe, filename, celldata, labels = ['Scalar']):
    if mpicomm.rank == 0:
        Points = vtk.vtkPoints()
        Triangles = vtk.vtkCellArray()
        
        for itri in range(universe.N):
            
            Triangle = vtk.vtkTriangle()
            id = Points.InsertNextPoint(universe.v1[itri][0], universe.v1[itri][1], universe.v1[itri][2])
            id = Points.InsertNextPoint(universe.v2[itri][0], universe.v2[itri][1], universe.v2[itri][2])
            id = Points.InsertNextPoint(universe.v3[itri][0], universe.v3[itri][1], universe.v3[itri][2])

            Triangle.GetPointIds().SetId(0, 3*itri)
            Triangle.GetPointIds().SetId(1, 3*itri+1)
            Triangle.GetPointIds().SetId(2, 3*itri+2)
            Triangles.InsertNextCell(Triangle)


        polydata = vtk.vtkPolyData()
        polydata.SetPoints(Points)
        polydata.SetPolys(Triangles)
        polydata.Modified()
        for index, data in enumerate(celldata):
            vtkdata = numpy_to_vtk(data)
            vtkdata.SetName(labels[index])
            polydata.GetCellData().AddArray(vtkdata)
        
            if vtk.VTK_MAJOR_VERSION <= 5:
                polydata.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()

def rprint(mpicomm, string):
    if mpicomm.rank == 0:
        print(string)


# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


class Comm:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def getcomm(self):
        return self.comm

    def getrank(self):
        return self.comm.Get_rank()

    def getsize(self):
        return self.comm.Get_size()



class Universe:
    def __init__(self):
        self.v1=[]
        self.v2=[]
        self.v3=[]
        self.x=[]
        self.y=[]
        self.z=[]
        self.normals=[]
        self.centers=[]
        self.areas=[]
        self.types=[]
        self.rhos=[]
        self.IDs=[]
        self.N=0

    def add(self, MpiComm, meshfile, ID, surftype, rho = 0):
        surfmesh = mesh.Mesh.from_file(meshfile)

        rank = MpiComm.rank
        
        self.v1.extend(surfmesh.v0)
        self.v2.extend(surfmesh.v1)
        self.v3.extend(surfmesh.v2)

        self.x.extend(surfmesh.x)
        self.y.extend(surfmesh.y)
        self.z.extend(surfmesh.z)
        
        for ntri in range(len(surfmesh.normals)):
            normal = surfmesh.normals[ntri]
            self.normals.append(normal / np.sqrt(np.sum(normal**2)))
            
            self.centers.append((surfmesh.v0[ntri] + surfmesh.v1[ntri] + surfmesh.v2[ntri]) / 3)

            if surftype == 'wall':
                rho = 1
                self.types.append(0)
            elif surftype == 'outlet':
                self.types.append(0)
            elif surftype == 'inlet':
                rho = 0
                self.types.append(1)
            else:
                self.types.append(-1)

            self.rhos.append(rho)

            self.IDs.append(ID)

            self.areas.append(0.5*np.linalg.norm(np.cross(surfmesh.v1[ntri] - surfmesh.v0[ntri], surfmesh.v2[ntri] - surfmesh.v0[ntri])))

        self.N += len(surfmesh.v0)
        
        if rank == 0:
            print('Added {numsurf} triangles from stl file'.format(numsurf=len(surfmesh.normals)))

    def init(self, mpicomm):
        self.centers=np.array(self.centers, dtype=np.float64)
        self.normals=np.array(self.normals, dtype=np.float64)
        self.areas=np.array(self.areas, dtype=np.float64)
        self.v1=np.array(self.v1, dtype=np.float64)
        self.v2=np.array(self.v2, dtype=np.float64)
        self.v3=np.array(self.v3, dtype=np.float64)

        self.chunks, self.offsets = split_proc(range(self.N), mpicomm.getsize())

        nIDs=max(self.IDs)+1
        nIDs=64
        self.n=[0.0 for i in range(nIDs)]
        self.T=[0.0 for i in range(nIDs)]
        self.S=[0.0 for i in range(nIDs)]
        self.uhat=[[0.0, 0.0, 0.0] for i in range(nIDs)]
        self.m=0.0


    def prop(self, ID, n = 0.0, T = 0.0, S = 0.0, uhat = [0.0, 0.0, 0.0], m = 0.0):
        self.n[ID]=n
        self.T[ID]=T
        self.S[ID]=S
        self.uhat[ID]=uhat
        self.m = m
