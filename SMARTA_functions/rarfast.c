#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

// Module method definitions

#define normals(i,j) (*(double*)((PyArray_DATA(py_normals)+(i)*PyArray_STRIDES(py_normals)[0]+(j)*PyArray_STRIDES(py_normals)[1])))

#define centers(i,j) (*(double*)((PyArray_DATA(py_centers)+(i)*PyArray_STRIDES(py_centers)[0]+(j)*PyArray_STRIDES(py_centers)[1])))

#define v1(i,j) (*(double*)((PyArray_DATA(py_v1)+(i)*PyArray_STRIDES(py_v1)[0]+(j)*PyArray_STRIDES(py_v1)[1])))

#define v2(i,j) (*(double*)((PyArray_DATA(py_v2)+(i)*PyArray_STRIDES(py_v2)[0]+(j)*PyArray_STRIDES(py_v2)[1])))

#define v3(i,j) (*(double*)((PyArray_DATA(py_v3)+(i)*PyArray_STRIDES(py_v3)[0]+(j)*PyArray_STRIDES(py_v3)[1])))

#define local_F(i,j) (*(double*)((PyArray_DATA(py_local_F)+(i)*PyArray_STRIDES(py_local_F)[0]+(j)*PyArray_STRIDES(py_local_F)[1])))

#define local_M(i,j) (*(double*)((PyArray_DATA(py_local_M)+(i)*PyArray_STRIDES(py_local_M)[0]+(j)*PyArray_STRIDES(py_local_M)[1])))

#define local_P(i,j) (*(double*)((PyArray_DATA(py_local_P)+(i)*PyArray_STRIDES(py_local_P)[0]+(j)*PyArray_STRIDES(py_local_P)[1])))

#define local_Ene(i,j) (*(double*)((PyArray_DATA(py_local_Ene)+(i)*PyArray_STRIDES(py_local_Ene)[0]+(j)*PyArray_STRIDES(py_local_Ene)[1])))

#define local_sx(i,j) (*(double*)((PyArray_DATA(py_local_sx)+(i)*PyArray_STRIDES(py_local_sx)[0]+(j)*PyArray_STRIDES(py_local_sx)[1])))
#define local_sy(i,j) (*(double*)((PyArray_DATA(py_local_sy)+(i)*PyArray_STRIDES(py_local_sy)[0]+(j)*PyArray_STRIDES(py_local_sy)[1])))
#define local_sz(i,j) (*(double*)((PyArray_DATA(py_local_sz)+(i)*PyArray_STRIDES(py_local_sz)[0]+(j)*PyArray_STRIDES(py_local_sz)[1])))

#define areas(i) (*(double*)((PyArray_DATA(py_areas)+(i)*PyArray_STRIDES(py_areas)[0])))


static inline void progress( char label[], int step, int total )
{
    float percent = ( step * 100.0f ) / total;
    printf("Progress: %3.1f%%\n", percent );
	fflush(stdout);
}


static inline void vector(double* result, double* v1, double* v2){
	result[0]=v2[0]-v1[0];
	result[1]=v2[1]-v1[1];
	result[2]=v2[2]-v1[2];
}

static inline double dot(double* v1, double* v2){
	return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}
	
static inline void cross(double* result, double* v1, double* v2){
	result[0] = v1[1]*v2[2]-v1[2]*v2[1];
	result[1] = v2[0]*v1[2]-v1[0]*v2[2];
	result[2] = v1[0]*v2[1]-v2[0]*v1[1];
}

static inline double veclen(double* v){
	return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

static inline double dist(double* P1, double* P2) {
	return sqrt((P2[0]-P1[0])*(P2[0]-P1[0])+(P2[1]-P1[1])*(P2[1]-P1[1])+(P2[2]-P1[2])*(P2[2]-P1[2]));
}

static inline double view_factor(int i, int j, PyArrayObject *py_areas, PyArrayObject *py_normals, PyArrayObject *py_centers, PyArrayObject *py_v1, PyArrayObject *py_v2, PyArrayObject *py_v3) {

	double Ai = areas(i);
	double Aj = areas(j);

	double factor = 400.0;
	double sij[3];
	sij[0]= centers(j,0)-centers(i,0);
	sij[1]= centers(j,1)-centers(i,1);
	sij[2]= centers(j,2)-centers(i,2);
	double sij2 = sij[0]*sij[0]+sij[1]*sij[1]+sij[2]*sij[2];
        double sijnorm = sqrt(sij2);
        sij[0]=sij[0]/sijnorm;
        sij[1]=sij[1]/sijnorm;
        sij[2]=sij[2]/sijnorm;
	if (sij2 > factor * Ai) {
		if (sij2 > factor * Aj) {
			double Fij = -(normals(i,0)*sij[0]+normals(i,1)*sij[1]+normals(i,2)*sij[2]) * (normals(j,0)*sij[0]+normals(j,1)*sij[1]+normals(j,2)*sij[2]) / M_PI / sij2;
			return Fij;
		}
	}

	int N = 20;

	double sum = 0;
	double K1[3];
	double K2[3];
	double L1[3];
	double L2[3];
	for (int k = 0; k < 3; k++){

		if (k==0) {
			K1[0] = v1(i,0);
			K1[1] = v1(i,1);
			K1[2] = v1(i,2);

			K2[0] = v2(i,0);
			K2[1] = v2(i,1);
			K2[2] = v2(i,2);
		} else if (k==1) {
			K1[0] = v2(i,0);
			K1[1] = v2(i,1);
			K1[2] = v2(i,2);

			K2[0] = v3(i,0);
			K2[1] = v3(i,1);
			K2[2] = v3(i,2);
		} else {
			K1[0] = v3(i,0);
			K1[1] = v3(i,1);
			K1[2] = v3(i,2);

			K2[0] = v1(i,0);
			K2[1] = v1(i,1);
			K2[2] = v1(i,2);
		}

		for (int l = 0; l < 3; l++){

			if (l==0) {
				L1[0] = v1(j,0);
				L1[1] = v1(j,1);
				L1[2] = v1(j,2);

				L2[0] = v2(j,0);
				L2[1] = v2(j,1);
				L2[2] = v2(j,2);
			} else if (l==1) {
				L1[0] = v2(j,0);
				L1[1] = v2(j,1);
				L1[2] = v2(j,2);

				L2[0] = v3(j,0);
				L2[1] = v3(j,1);
				L2[2] = v3(j,2);
			} else {
				L1[0] = v3(j,0);
				L1[1] = v3(j,1);
				L1[2] = v3(j,2);

				L2[0] = v1(j,0);
				L2[1] = v1(j,1);
				L2[2] = v1(j,2);
			}

			double integral = 0;
			double K[3];
			vector(K, K1, K2);
			double L[3];
			vector(L, L1, L2);
			double lenk = veclen(K);
			double lenl = veclen(L);
			double Dkl = dot(K, L)/lenk/lenl;
			double dk = lenk/N;


			for (int nk = 0; nk < N; nk++) {
				double K[3];
				K[0] = K1[0] + ((double) nk + 0.5)*(K2[0]-K1[0])/N;
				K[1] = K1[1] + ((double) nk + 0.5)*(K2[1]-K1[1])/N;
				K[2] = K1[2] + ((double) nk + 0.5)*(K2[2]-K1[2])/N;

				double KL1[3];
				vector(KL1, K, L1);

				double KL2[3];
				vector(KL2, K, L2);
				double Lalpha = veclen(KL1);
				double Lbeta = veclen(KL2);
				
				double cosgamma = (Lalpha*Lalpha+Lbeta*Lbeta-lenl*lenl)/(2*Lalpha*Lbeta);
				double gamma;
				if (cosgamma <= -1) {
					gamma = M_PI;
				} else if (cosgamma >= 1) {
					gamma = 0.0;
				} else {
					gamma = acos((Lalpha*Lalpha+Lbeta*Lbeta-lenl*lenl)/(2*Lalpha*Lbeta));
				}
				
				double cosalpha = -dot(L, KL1)/lenl/Lalpha;
				double cosbeta = dot(L, KL2)/lenl/Lbeta;

				double pcross[3];
				cross(pcross, KL1, KL2);
				double P = veclen(pcross) / lenl;

				double dint = (Lalpha*log(Lalpha)*cosalpha + Lbeta*log(Lbeta)*cosbeta + P*gamma - lenl) * dk;
				integral += dint;
			}
			integral *= Dkl;
			sum += integral;
		}
	}

	if (sum < 0){
		sum = 0.0;
	}

	return sum / (M_PI * 2.0 * Ai * Aj);
}

static inline int inview_c(int N, long i, int j, PyArrayObject *py_centers, PyArrayObject *py_normals, PyArrayObject *py_v1, PyArrayObject *py_v2, PyArrayObject *py_v3) {

	double epsilon = 1e-6;

	double sij[3];
	sij[0]= centers(j,0)-centers(i,0);
	sij[1]= centers(j,1)-centers(i,1);
	sij[2]= centers(j,2)-centers(i,2);
	//printf("sij %f, %f, %f \n", sij[0], sij[1], sij[2]);

	double doti = normals(i,0)*sij[0]+normals(i,1)*sij[1]+normals(i,2)*sij[2];
	double dotj = normals(j,0)*sij[0]+normals(j,1)*sij[1]+normals(j,2)*sij[2];


	if (doti < epsilon || dotj > -epsilon){ // triangles do not face each other
		return 0;
	}

	for (int k = 0; k < N; k++){
		if (k == i || k == j) {
	    continue;
		}
		double nkdotsij = normals(k,0)*sij[0]+normals(k,1)*sij[1]+normals(k,2)*sij[2];
		if (nkdotsij < epsilon && nkdotsij > -epsilon) { // Without back-face culling (symmetric visibility)
		  continue;
		}

		double sik[3];
		sik[0]= centers(k,0)-centers(i,0);
		sik[1]= centers(k,1)-centers(i,1);
		sik[2]= centers(k,2)-centers(i,2);
		double beta = (normals(k,0)*sik[0]+normals(k,1)*sik[1]+normals(k,2)*sik[2]) / nkdotsij;
		if (beta > (1.0-epsilon) || beta < epsilon) { // obstruction is not in between
			continue;
		}
		//printf("beta is: %f \n", beta);
		int jj, kk;
		double Nx = normals(k, 0);
		if (Nx < 0) {
			Nx = -Nx;
		}
		double Ny = normals(k, 1);
		if (Ny < 0) {
			Ny = -Ny;
		}
		double Nz = normals(k, 2);
		if (Nz < 0) {
			Nz = -Nz;
		}
		if (Nx > Ny){
			if (Nx > Nz){
				jj=1;
				kk=2;
			} else {
				jj=0;
				kk=1;
			}
		} else {
			if (Ny < Nz){
				jj=0;
				kk=1;
			}	else {
				jj=0;
				kk=2;
			}
		}
		double Pj = centers(i,jj) + sij[jj]*beta;
		double Pk = centers(i,kk) + sij[kk]*beta;
		double U1 = Pj-v1(k, jj);
		double V1 = Pk-v1(k, kk);
		double U2 = v2(k, jj) - v1(k, jj);
		double U3 = v3(k, jj) - v1(k, jj);
		double V2 = v2(k, kk) - v1(k, kk);
		double V3 = v3(k, kk) - v1(k, kk);
		
		double a, b;
		if (U2 > -epsilon && U2 < epsilon){
			b = U1/U3;
			if (b < 0.0 || b > 1.0) {
				continue;
			}
			a = (V1 - b * V3) / V2;
		} else {
			b = (V1 * U2 - U1 * V2) / (V3 * U2 - U3 * V2);
			if (b < 0.0 || b > 1.0) {
				continue;
			}
			a = (U1 - b * U3) / U2;
		}
		if (a < 0.0 || (a + b) > 1.0) {
			continue;
		}
		return 0;
	}
	return 1;
}






static inline double Mij(double S, double cosalpha) {
  double sig = S*cosalpha;
  double phi = sqrt(M_PI) * sig * exp(S*S * (cosalpha*cosalpha-1.0)) * (sig*sig + 1.5) * (erf(sig)+1.0) + exp(-S*S) * (sig*sig+1.0);
  return phi;
}


static inline double Pij(double S, double cosalpha) {
  double sig = S*cosalpha;
  double phi = exp(S*S * (cosalpha*cosalpha-1.0)) * (4.0/3.0*pow(sig, 4.0) + 4.0*sig*sig + 1.0) * (erf(sig)+1.0) + exp(-S*S) * sig/sqrt(M_PI)*(4.0/3.0*sig*sig+10.0/3.0);
  return phi;
}


static inline double Eneij(double S, double cosalpha) {
  double sig = S*cosalpha;
  double phi = exp(S*S * (cosalpha*cosalpha-1.0)) * sqrt(M_PI) * sig * (0.5*pow(sig, 4.0) + 2.5*sig*sig + 15.0/8.0) * (erf(sig)+1.0) + exp(-S*S) * (2.0*sig*sig+1)*(0.25*sig*sig+1);
  return phi;
}



static PyObject* compute_F_c(PyObject *dummy, PyObject *args) {

PyArrayObject *py_local_F;
PyArrayObject *py_centers;
PyArrayObject *py_normals;
PyArrayObject *py_v1;
PyArrayObject *py_v2;
PyArrayObject *py_v3;
PyArrayObject *py_areas;
PyObject *mychunk;
int N;


if (!PyArg_ParseTuple(args, "lOO!O!O!O!O!O!O!", &N, &mychunk, &PyArray_Type, &py_local_F, &PyArray_Type, &py_centers, &PyArray_Type, &py_normals, &PyArray_Type, &py_v1, &PyArray_Type, &py_v2, &PyArray_Type, &py_v3, &PyArray_Type, &py_areas))
    return NULL;

PyObject *mychunkiter = PyObject_GetIter(mychunk);
int ii = 0;

while(1) {

  PyObject *next = PyIter_Next(mychunkiter);

  if (!next) {
      // nothing left in the iterator
      break;
  }
  long i = PyLong_AsLong(next);
  
  if (i % 100 == 0){
	progress("F matrix", i, N);
  } 
  
  for (int j = 0; j < (i+1); j++) {
    if (i!=j) {
      if (inview_c(N, i, j, py_centers, py_normals, py_v1, py_v2, py_v3)) {
        local_F(ii,j) = view_factor(i, j, py_areas, py_normals, py_centers, py_v1, py_v2, py_v3);
      }
    }
  }
  
  ii++;
}

Py_RETURN_NONE;
}




static PyObject* compute_M_c(PyObject *dummy, PyObject *args) {


PyArrayObject *py_local_M;
PyArrayObject *py_centers;
PyObject *mychunk;
PyObject *IDs;
PyObject *types;
PyObject *uhat;
PyObject *Ss;
int N;


if (!PyArg_ParseTuple(args, "lOOOOOO!O!", &N, &mychunk, &IDs, &types, &uhat, &Ss, &PyArray_Type, &py_local_M, &PyArray_Type, &py_centers))
    return NULL;


PyObject *mychunkiter = PyObject_GetIter(mychunk);
int ii = 0;

while(1) {

  PyObject *next = PyIter_Next(mychunkiter);

  if (!next) {
      // nothing left in the iterator
      break;
  }
  long i = PyLong_AsLong(next);
  if (i % 100 == 0){
	progress("M matrix", i, N);
  } 
  
  if (PyLong_AsLong(PyList_GetItem(types, i)) == 1) {
    for (int j = 0; j < N; j++) {
      if (i!=j) {  
        double sij[3];
        sij[0]= centers(j,0)-centers(i,0);
        sij[1]= centers(j,1)-centers(i,1);
        sij[2]= centers(j,2)-centers(i,2);
        double sij2 = sij[0]*sij[0]+sij[1]*sij[1]+sij[2]*sij[2];
        double sijnorm = sqrt(sij2);
        sij[0]=sij[0]/sijnorm;
        sij[1]=sij[1]/sijnorm;
        sij[2]=sij[2]/sijnorm;
        
        long ID = PyLong_AsLong(PyList_GetItem(IDs, i));

        PyObject *thisuhat = PyList_GetItem(uhat, ID);

        double cuhat[3];
        cuhat[0] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 0));
        cuhat[1] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 1));
        cuhat[2] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 2));

        double cosalpha = cuhat[0]*sij[0]+cuhat[1]*sij[1]+cuhat[2]*sij[2];
        double S = PyFloat_AsDouble(PyList_GetItem(Ss, ID));
        local_M(ii,j) = Mij(S, cosalpha);
      }
    }
  }
  
  ii++;
}

Py_RETURN_NONE;
}




static PyObject* compute_P_c(PyObject *dummy, PyObject *args) {


PyArrayObject *py_local_P;
PyArrayObject *py_centers;
PyObject *mychunk;
PyObject *IDs;
PyObject *types;
PyObject *uhat;
PyObject *Ss;
int N;


if (!PyArg_ParseTuple(args, "lOOOOOO!O!", &N, &mychunk, &IDs, &types, &uhat, &Ss, &PyArray_Type, &py_local_P, &PyArray_Type, &py_centers))
    return NULL;


PyObject *mychunkiter = PyObject_GetIter(mychunk);
int ii = 0;

while(1) {

  PyObject *next = PyIter_Next(mychunkiter);

  if (!next) {
      // nothing left in the iterator
      break;
  }
  long i = PyLong_AsLong(next);

  if (i % 100 == 0){
	progress("P matrix", i, N);
  } 
  
  if (PyLong_AsLong(PyList_GetItem(types, i)) == 1) {
    for (int j = 0; j < N; j++) {
      if (i!=j) {  
        double sij[3];
        sij[0]= centers(j,0)-centers(i,0);
        sij[1]= centers(j,1)-centers(i,1);
        sij[2]= centers(j,2)-centers(i,2);
        double sij2 = sij[0]*sij[0]+sij[1]*sij[1]+sij[2]*sij[2];
        double sijnorm = sqrt(sij2);
        sij[0]=sij[0]/sijnorm;
        sij[1]=sij[1]/sijnorm;
        sij[2]=sij[2]/sijnorm;
        
        long ID = PyLong_AsLong(PyList_GetItem(IDs, i));
        
        PyObject *thisuhat = PyList_GetItem(uhat, ID);
        
        double cuhat[3];
        cuhat[0] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 0));
        cuhat[1] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 1));
        cuhat[2] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 2));
        
        double cosalpha = cuhat[0]*sij[0]+cuhat[1]*sij[1]+cuhat[2]*sij[2];
        double S = PyFloat_AsDouble(PyList_GetItem(Ss, ID));
        local_P(ii,j) = Pij(S, cosalpha);
      }
    }
  } else if (PyLong_AsLong(PyList_GetItem(types, i)) == 0) {
    for (int j = 0; j < N; j++) {
      if (i!=j) {  
        local_P(ii,j) = 1.0;
      }
    }
  }
  
  ii++;
}

Py_RETURN_NONE;
}




static PyObject* compute_s_c(PyObject *dummy, PyObject *args) {

PyArrayObject *py_local_sx;
PyArrayObject *py_local_sy;
PyArrayObject *py_local_sz;
PyArrayObject *py_centers;
PyObject *mychunk;
int N;


if (!PyArg_ParseTuple(args, "lOO!O!O!O!", &N, &mychunk, &PyArray_Type, &py_centers, &PyArray_Type, &py_local_sx, &PyArray_Type, &py_local_sy, &PyArray_Type, &py_local_sz))
    return NULL;

PyObject *mychunkiter = PyObject_GetIter(mychunk);
int ii = 0;

while(1) {

  PyObject *next = PyIter_Next(mychunkiter);

  if (!next) {
      // nothing left in the iterator
      break;
  }
  long i = PyLong_AsLong(next);

  if (i % 100 == 0){
	progress("s matrix", i, N);
  } 

  for (int j = 0; j < (i+1); j++) {
    if (i!=j) {
      double sij[3];
      sij[0]= centers(j,0)-centers(i,0);
      sij[1]= centers(j,1)-centers(i,1);
      sij[2]= centers(j,2)-centers(i,2);
      double sij2 = sij[0]*sij[0]+sij[1]*sij[1]+sij[2]*sij[2];
      double sijnorm = sqrt(sij2);
      sij[0]=sij[0]/sijnorm;
      sij[1]=sij[1]/sijnorm;
      sij[2]=sij[2]/sijnorm;
 
      local_sx(ii,j) = sij[0];
      local_sy(ii,j) = sij[1];
      local_sz(ii,j) = sij[2];
      
    }
  }
  
  ii++;
}

Py_RETURN_NONE;
}





static PyObject* compute_Ene_c(PyObject *dummy, PyObject *args) {


PyArrayObject *py_local_Ene;
PyArrayObject *py_centers;
PyObject *mychunk;
PyObject *IDs;
PyObject *types;
PyObject *uhat;
PyObject *Ss;
int N;


if (!PyArg_ParseTuple(args, "lOOOOOO!O!", &N, &mychunk, &IDs, &types, &uhat, &Ss, &PyArray_Type, &py_local_Ene, &PyArray_Type, &py_centers))
    return NULL;


PyObject *mychunkiter = PyObject_GetIter(mychunk);
int ii = 0;

while(1) {

  PyObject *next = PyIter_Next(mychunkiter);

  if (!next) {
      // nothing left in the iterator
      break;
  }
  long i = PyLong_AsLong(next);

  if (i % 100 == 0){
	progress("Ene matrix", i, N);
  } 
  
  if (PyLong_AsLong(PyList_GetItem(types, i)) == 1) {
    for (int j = 0; j < N; j++) {
      if (i!=j) {  
        double sij[3];
        sij[0]= centers(j,0)-centers(i,0);
        sij[1]= centers(j,1)-centers(i,1);
        sij[2]= centers(j,2)-centers(i,2);
        double sij2 = sij[0]*sij[0]+sij[1]*sij[1]+sij[2]*sij[2];
        double sijnorm = sqrt(sij2);
        sij[0]=sij[0]/sijnorm;
        sij[1]=sij[1]/sijnorm;
        sij[2]=sij[2]/sijnorm;
        
        long ID = PyLong_AsLong(PyList_GetItem(IDs, i));
        
        PyObject *thisuhat = PyList_GetItem(uhat, ID);
        
        double cuhat[3];
        cuhat[0] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 0));
        cuhat[1] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 1));
        cuhat[2] = PyFloat_AsDouble(PyList_GetItem(thisuhat, 2));
        
        double cosalpha = cuhat[0]*sij[0]+cuhat[1]*sij[1]+cuhat[2]*sij[2];
        double S = PyFloat_AsDouble(PyList_GetItem(Ss, ID));
        local_Ene(ii,j) = Eneij(S, cosalpha);
      }
    }
  } else if (PyLong_AsLong(PyList_GetItem(types, i)) == 0) {
    for (int j = 0; j < N; j++) {
      if (i!=j) {  
        local_Ene(ii,j) = 1.0;
      }
    }
  }
  
  ii++;
}

Py_RETURN_NONE;
}





static PyMethodDef rarfast_methods[] = {
        {
                "compute_F", compute_F_c, METH_VARARGS,
                "View Factor Matrix Computation",
        },
        {
                "compute_M", compute_M_c, METH_VARARGS,
                "M Matrix Computation",
        },
        {
                "compute_P", compute_P_c, METH_VARARGS,
                "P Matrix Computation",
        },
        {
                "compute_s", compute_s_c, METH_VARARGS,
                "s Matrix Computation",
        },
        {
                "compute_Ene", compute_Ene_c, METH_VARARGS,
                "Ene Matrix Computation",
        },
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef rarfast_definition = {
        PyModuleDef_HEAD_INIT,
        "rarfast",
        "RAREFIED extension module using C API.",
        -1,
        rarfast_methods
};


PyMODINIT_FUNC PyInit_rarfast(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&rarfast_definition);
}
