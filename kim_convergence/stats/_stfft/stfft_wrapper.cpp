/*
 * Single-threaded FFT wrapper using pocketfft
 *
 * This module provides single-threaded FFT to avoid threading deadlocks
 * in multi-process simulation environments (e.g., LAMMPS with MPI).
 * Compiled WITHOUT OpenMP.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Disable threading in pocketfft
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include <vector>
#include <complex>
#include <algorithm>
#include <new>
#include <stdexcept>

using namespace pocketfft;

/*
 * Real FFT (rfft) - single-threaded
 */
static PyObject* stfft_rfft(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject *input = NULL;
    PyObject *n_obj = NULL;
    Py_ssize_t n = 0;

    static char *kwlist[] = {(char*)"x", (char*)"n", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist,
                                     &PyArray_Type, &input, &n_obj)) {
        return NULL;
    }

    if (PyArray_NDIM(input) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1-dimensional");
        return NULL;
    }

    npy_intp input_len = PyArray_DIM(input, 0);
    if (n_obj == NULL || n_obj == Py_None) {
        n = input_len;
    } else {
        n = PyLong_AsSsize_t(n_obj);
        if (n == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }

    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }

    PyArrayObject *input_double = (PyArrayObject*)PyArray_Cast(input, NPY_DOUBLE);
    if (input_double == NULL) {
        return NULL;
    }

    if (!PyArray_ISCONTIGUOUS(input_double)) {
        PyArrayObject *contiguous = (PyArrayObject*)PyArray_ContiguousFromAny(
            (PyObject*)input_double, NPY_DOUBLE, 1, 1);
        Py_DECREF(input_double);
        if (contiguous == NULL) {
            return NULL;
        }
        input_double = contiguous;
    }

    double *data = (double*)PyArray_DATA(input_double);

    PyArrayObject *output = NULL;

    try {
        std::vector<double> in_data(n, 0.0);
        for (Py_ssize_t i = 0; i < std::min((Py_ssize_t)input_len, n); i++) {
            in_data[i] = data[i];
        }

        size_t out_len = n / 2 + 1;
        std::vector<std::complex<double>> out_data(out_len);

        shape_t shape{(size_t)n};
        shape_t axes{0};
        stride_t stride_in{sizeof(double)};
        stride_t stride_out{sizeof(std::complex<double>)};

        r2c(shape, stride_in, stride_out, axes, FORWARD,
            in_data.data(), out_data.data(), 1.0, 1);

        npy_intp dims[1] = {(npy_intp)out_len};
        output = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
        if (output == NULL) {
            Py_DECREF(input_double);
            return NULL;
        }

        std::complex<double> *out_ptr = (std::complex<double>*)PyArray_DATA(output);
        for (size_t i = 0; i < out_len; i++) {
            out_ptr[i] = out_data[i];
        }
    } catch (const std::bad_alloc &) {
        Py_XDECREF(output);
        Py_DECREF(input_double);
        PyErr_NoMemory();
        return NULL;
    } catch (const std::exception &e) {
        Py_XDECREF(output);
        Py_DECREF(input_double);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        Py_XDECREF(output);
        Py_DECREF(input_double);
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception in rfft");
        return NULL;
    }

    Py_DECREF(input_double);
    return (PyObject*)output;
}

/*
 * Inverse real FFT (irfft) - single-threaded
 */
static PyObject* stfft_irfft(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject *input = NULL;
    PyObject *n_obj = NULL;
    Py_ssize_t n = 0;

    static char *kwlist[] = {(char*)"x", (char*)"n", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O", kwlist,
                                     &PyArray_Type, &input, &n_obj)) {
        return NULL;
    }

    if (PyArray_NDIM(input) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1-dimensional");
        return NULL;
    }

    npy_intp input_len = PyArray_DIM(input, 0);
    if (input_len == 0) {
        PyErr_SetString(PyExc_ValueError, "Input array cannot be empty");
        return NULL;
    }

    // Internal call paths pass n explicitly; this default branch is for direct wrapper calls.
    if (n_obj == NULL || n_obj == Py_None) {
        n = (input_len - 1) * 2;
    } else {
        n = PyLong_AsSsize_t(n_obj);
        if (n == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }

    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }

    PyArrayObject *input_complex = (PyArrayObject*)PyArray_Cast(input, NPY_COMPLEX128);
    if (input_complex == NULL) {
        return NULL;
    }

    if (!PyArray_ISCONTIGUOUS(input_complex)) {
        PyArrayObject *contiguous = (PyArrayObject*)PyArray_ContiguousFromAny(
            (PyObject*)input_complex, NPY_COMPLEX128, 1, 1);
        Py_DECREF(input_complex);
        if (contiguous == NULL) {
            return NULL;
        }
        input_complex = contiguous;
    }

    std::complex<double> *data = (std::complex<double>*)PyArray_DATA(input_complex);

    PyArrayObject *output = NULL;

    try {
        size_t expected_len = n / 2 + 1;
        std::vector<std::complex<double>> in_data(expected_len, std::complex<double>(0.0, 0.0));

        for (Py_ssize_t i = 0; i < std::min((Py_ssize_t)input_len, (Py_ssize_t)expected_len); i++) {
            in_data[i] = data[i];
        }

        std::vector<double> out_data(n);

        shape_t shape{(size_t)n};
        shape_t axes{0};
        stride_t stride_in{sizeof(std::complex<double>)};
        stride_t stride_out{sizeof(double)};

        c2r(shape, stride_in, stride_out, axes, BACKWARD,
            in_data.data(), out_data.data(), 1.0 / n, 1);

        npy_intp dims[1] = {(npy_intp)n};
        output = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (output == NULL) {
            Py_DECREF(input_complex);
            return NULL;
        }

        double *out_ptr = (double*)PyArray_DATA(output);
        for (size_t i = 0; i < (size_t)n; i++) {
            out_ptr[i] = out_data[i];
        }
    } catch (const std::bad_alloc &) {
        Py_XDECREF(output);
        Py_DECREF(input_complex);
        PyErr_NoMemory();
        return NULL;
    } catch (const std::exception &e) {
        Py_XDECREF(output);
        Py_DECREF(input_complex);
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (...) {
        Py_XDECREF(output);
        Py_DECREF(input_complex);
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception in irfft");
        return NULL;
    }

    Py_DECREF(input_complex);
    return (PyObject*)output;
}

static PyMethodDef stfft_methods[] = {
    {"rfft", (PyCFunction)stfft_rfft, METH_VARARGS | METH_KEYWORDS,
     "Real FFT (single-threaded)"},
    {"irfft", (PyCFunction)stfft_irfft, METH_VARARGS | METH_KEYWORDS,
     "Inverse real FFT (single-threaded)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef stfft_module = {
    PyModuleDef_HEAD_INIT,
    "_stfft_core",
    "Single-threaded FFT using pocketfft",
    -1,
    stfft_methods
};

PyMODINIT_FUNC PyInit__stfft_core(void) {
    import_array();
    return PyModule_Create(&stfft_module);
}
