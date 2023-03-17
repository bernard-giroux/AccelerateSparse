# -*- coding: utf-8 -*-
"""
Python interface to the Sparse functions of Apple's Accelerate Framework

This module contains a class for solving Ax=b systems with direct solvers

Ultimately, iterative solvers will be added.
"""
# distutils: language = c++

import warnings

from libcpp cimport bool
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc, tril, triu

from pyAccelerate.cSolve cimport SparseKind_t,  SparseStatus_t, SparseFactorization_t, \
SparseTriangle_t, SparseMatrix_Double, DenseVector_Double, \
SparseOpaqueFactorization_Double, SparseFactor, SparseRefactor, SparseSolve, SparseCleanup


def is_symmetric(A):
    i = np.finfo(A.dtype)
    d = tril(A) - triu(A).T
    return np.all(np.abs(d.data) < 10*i.eps)


cdef class DirectSolver:
    """
    Class for solving Ax=b systems with direct solvers

    Attributes
    ----------
    verbose: bool
        print informative messages if True

    Constructor:

    DirectSolver(verbose=False)

        Parameters
        ----------
        verbose : bool
            set verbosity

    Notes
    -----

    The following is taken and adapted from Solve.h in the Accelerate framwork

    ======================================
    Direct Methods (Matrix Factorizations)
    ======================================

    We offer the factorizations detailed below, but all use the same interface,
    with the variant specified by the argument `type`. The most basic solution
    sequence is:
      factors = SparseFactor(type, Matrix)
      SparseSolve(factors, rhs, soln)
      SparseCleanup(factors)

    It is sometimes required to perform repeated factorizations with the same
    non-zero pattern but different numerical values. A SparseRefactor() entry
    point is supplied that allows the reuse of an existing factor object and
    its associated memory with different numerical values.

    If multiple different numeric factorizations with the same symbolic pattern
    are required, or if the user wishes to perform ordering before numeric
    values are known, a symbolic factorization object can be obtained by passing
    a SparseMatrixStructure object to SparseFactor() in place of the full
    SparseMatrix that also includes the numeric values. The underlying object
    is reference counted, so this object may be safely destroyed by calling
    SparseOpaqueDestroy() even if numeric factorizations that depend on it are
    still in use. Due to this reference counting, if the user wishes to make a
    shallow copy of the underlying object they should call SparseRetain().

    If the user wishes to apply matrix factors individually, they may obtain
    opaque objects through the SparseCreateSubfactor() routine. These objects
    may then be used through calls to SparseMultiply() and SparseSove().

    Cholesky
    ========
    A = PLL'P'
    for real symmetric or complex Hermitian positive-definite matrices A.
    If A is not positive-definite the factorization will detect this and fail,
    potentially after significant computation.
    P is a permutation matrix that is (by default) automatically calculated by
    the solver (see options.orderMethod for further details).
    L is the lower triangular factorization matrix.
    By default no diagonal scaling matrix is applied to A, but one may be
    enabled through options.scalingMethod.

    SparseSolve() will solve Ax = b.
    SparseCreateSubfactor() allows the following subfactors to be extracted:
    - SparseSubfactorL    returns an opaque object representing L. Both Multiply and Solve are valid.
    - SparseSubfactorP    returns an opaque object representing P. Both Multiply and Solve are valid.
    - SparseSubfactorPLPS returns an opaque object representing PLP'. Only Solve is valid, and
                          transpose solve followed by non-transpose solve is equivalent to a full
                          system solve with A.

    Symmetric Indefinite
    ====================
    SAS = PLDL'P'
    for real symmetric or complex Hermitian matrices A.
    P is a permutation matrix that is (by default) automatically calculated by
    the solver (see options.orderMethod for further details).
    S is a diagonal scaling matrix that is (by default) automatically calculated
    by the solver (see options.scalingMethod for further details).
    L is a unit lower triangular factorization matrix.
    D is a block diagonal factorization matrix, with 1x1 and 2x2 diagonal blocks.
    A variety of different pivoting options are offered:
    - Unpivoted performs no numerical pivoting, and D only has 1x1 pivots. Only
      suitable for well behaved systems with full rank, otherwise very unstable.
    - Supernode Bunch-Kaufmann (SBK) restricts pivoting to operations that do not
      alter the symbolic structure of the factors. Static pivoting (the addition
      (of sqrt(eps) to small diagonal entries) is used in the presence of small
      pivots. This method is often effective for well scaled matrices, but is
      not numerically stable for some systems.
    - Threshold Partial Pivoting (TPP) is provably numerically stable, but at the
      cost of (potentially) increased factor size and number of operations.

    SparseSolve() will solve Ax = b.
    SparseCreateSubfactor() allows the following sunfactors to be extracted:
    - SparseSubfactorL    returns an opaque object representing L. Both Multiply and Solve are valid.
    - SparseSubfactorD    returns an opaque object representing D. Both Multiply and Solve are valid.
    - SparseSubfactorP    returns an opaque object representing P. Both Multiply and Solve are valid.
    - SparseSubfactorS    returns an opaque object representing S. Both Multiply and Solve are valid.
    - SparseSubfactorPLPS returns an opaque object representing PLP'S. When tranposed represents PLDP'S.
                       Only Solve is valid, and transpose solve followed by non-transpose solve is
                       equivalent to a full system solve with A.

    QR
    ==
    A = QRP      if m >= n so A is overdetermined or square
    A = P'R'Q'   if m <  n so A is underdetermined
    for real or complex matrices A of size m x n.
    P is a column permutation that is (by default) automatically calculated by
    the solver (see options.orderMethod for further details).
    Q is an m x n (or n x m if underdetermined) orthagonal factor matrix.
    R is an n x n (or m x m if underdetermined) upper triangular factor matrix.

    If a Cholesky factorization of A^T A is desired (being the factor R) consider
    using the CholeskyAtA options below instead. This performs the same factorization but
    without the overhead of storing the Q factor.

    We note that in many cases other methods of solving a given problem are normally faster
    than the use of a Sparse QR factorization:
    - For least squares, use a dedicated least squares solution method
      (e.g. Diagonally preconditioned LSMR).
    - If a low rank approximation is required, multiply rank+5 random vectors by A and
      perform a dense QR of the result.

    SparseSolve() will solve either:
    - x = arg min_x || Ax - b ||_2      if A is overdetermined.
    - x = arg min_x || x ||_2 s.t. Ax=b if A is underdetermined.
    SparseCreateSubfactor() allows the following sunfactors to be extracted:
    - SparseSubfactorQ  returns an opaque object representing Q. Both Multiply and Solve are valid.
    - SparseSubfactorR  returns an opaque object representing R. Both Multiply and Solve are valid.
    - SparseSubfactorP  returns an opaque object representing P. Both Multiply and Solve are valid.
    - SparseSubfactorRP returns an opaque object representing RP (or P'R'). Only Solve is valid.

    CholeskyAtA
    ===========
    A^TA = P'R'RP
    for real matrices A.
    This performs the same factorization as QR above, but avoids storing the Q factor resulting
    in a significant storage saving. The number of rows in A must be greater than or equal to the
    number of columns (otherwise A^T A is singular).

    SparseSolve() will solve A^TA x = b.
    SparseCreateSubfactor() allows the following subfactors to be extracted:
    - SparseSubfactorR  returns an opaque object representing R. Both Multiply and Solve are valid.
    - SparseSubfactorP  returns an opaque object representing P. Both Multiply and Solve are valid.
   - SparseSubfactorRP returns an opaque object representing RP (or P'R'). Only Solve is valid.

    """
    cdef SparseOpaqueFactorization_Double factors
    cdef int row_count
    cdef int col_count
    cdef bool verbose
    
    def __cinit__(self, verbose=False):
        self.verbose = verbose

    def __del__(self):
        SparseCleanup(self.factors)

    def factorize(self, A, factorization_type='QR'):
        """
        Factorize matrix and store factors for further use.

        Parameters
        ----------
        A : spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
            matrix to factorize, will be converted to csc if in another format
        type : string
            type of factorization, choices are
            - Cholesky
            - LDLT
            - LDLTUnpivoted
            - LDLTSBK
            - LDLTTPP
            - QR
            - CholeskyAtA

        """
        # In the Accelerate framework, the Sparse Solvers library stores sparse matrices using the 
        # compressed sparse column (CSC) format
        if not isspmatrix_csc(A):
            A = csc_matrix(A)

        if A.dtype != np.float64:
            raise TypeError('Matrix data must by float64')

        cdef SparseMatrix_Double mat
        cdef SparseFactorization_t type_fact

        mat.structure.rowCount = self.row_count = A.shape[0]
        mat.structure.columnCount = self.col_count = A.shape[1]
        mat.structure.blockSize = 1
        
        symmetric_A = is_symmetric(A)
        if symmetric_A:
            if self.verbose:
                print('Symmetric matrix detected, working with lower triangular diagonal')
            A = tril(A, format='csc')
            mat.structure.attributes.kind = SparseKind_t.SparseSymmetric
            mat.structure.attributes.triangle = SparseTriangle_t.SparseLowerTriangle
        else:
            mat.structure.attributes.kind = SparseKind_t.SparseOrdinary


        if factorization_type == 'CholeskyAtA':
            type_fact = SparseFactorization_t.SparseFactorizationCholeskyAtA
        elif factorization_type == 'QR':
            if not symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationQR
            else:
                warnings.warn('Matrix is symmetric, using Cholesky factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationCholesky
        elif factorization_type == 'Cholesky':
            if symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationCholesky
            else:
                warnings.warn('Matrix not symmetric, using QR factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationQR
        elif factorization_type == 'LDLT':
            if symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationLDLT
            else:
                warnings.warn('Matrix not symmetric, using QR factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationQR
        elif factorization_type == 'LDLTUnpivoted':
            if symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationLDLTUnpivoted
            else:
                warnings.warn('Matrix not symmetric, using QR factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationQR
        elif factorization_type == 'LDLTSBK':
            if symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationLDLTSBK
            else:
                warnings.warn('Matrix not symmetric, using QR factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationQR
        elif factorization_type == 'LDLTTPP':
            if symmetric_A:
                type_fact = SparseFactorization_t.SparseFactorizationLDLTTPP
            else:
                warnings.warn('Matrix not symmetric, using QR factorization', RuntimeWarning)
                type_fact = SparseFactorization_t.SparseFactorizationQR
        else:
            raise ValueError('Factorization type is unknown')

        try:
            mat.structure.rowIndices = <int *> malloc(A.indices.size * sizeof(int))
            mat.structure.columnStarts = <long *> malloc(A.indptr.size * sizeof(long))
            mat.data = <double *> malloc(A.data.size * sizeof(double))

            for n in range(A.indices.size):
                mat.structure.rowIndices[n] = A.indices[n]
            for n in range(A.indptr.size):
                mat.structure.columnStarts[n] = A.indptr[n]
            for n in range(A.data.size):
                mat.data[n] = A.data[n]

            self.factors = SparseFactor(type_fact, mat)
            if self.factors.status != SparseStatus_t.SparseStatusOK:
                warnings.warn('Factorization failed with code '+str(self.factors.status), RuntimeWarning)

        finally:
            free(mat.structure.rowIndices)
            free(mat.structure.columnStarts)
            free(mat.data)

    def refactor(self, A):
        """
        Reuses factorization object's to compute a new factorization

        Parameters
        ----------
        A : spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)
            matrix to factorize, will be converted to csc if in another format
        """
        if not isspmatrix_csc(A):
            A = csc_matrix(A)

        if A.dtype != np.float64:
            raise TypeError('Matrix data must by float64')

        cdef SparseMatrix_Double mat

        mat.structure.rowCount = self.row_count = A.shape[0]
        mat.structure.columnCount = self.col_count = A.shape[1]
        mat.structure.blockSize = 1
        
        symmetric_A = is_symmetric(A)
        if symmetric_A:
            if self.verbose:
                print('Symmetric matrix detected, working with lower triangular diagonal')
            A = tril(A, format='csc')
            mat.structure.attributes.kind = SparseKind_t.SparseSymmetric
            mat.structure.attributes.triangle = SparseTriangle_t.SparseLowerTriangle
        else:
            mat.structure.attributes.kind = SparseKind_t.SparseOrdinary

        try:
            mat.structure.rowIndices = <int *> malloc(A.indices.size * sizeof(int))
            mat.structure.columnStarts = <long *> malloc(A.indptr.size * sizeof(long))
            mat.data = <double *> malloc(A.data.size * sizeof(double))

            for n in range(A.indices.size):
                mat.structure.rowIndices[n] = A.indices[n]
            for n in range(A.indptr.size):
                mat.structure.columnStarts[n] = A.indptr[n]
            for n in range(A.data.size):
                mat.data[n] = A.data[n]

            SparseRefactor(mat, &(self.factors))
            if self.factors.status != SparseStatus_t.SparseStatusOK:
                warnings.warn('Factorization failed with code '+str(self.factors.status), RuntimeWarning)

        finally:
            free(mat.structure.rowIndices)
            free(mat.structure.columnStarts)
            free(mat.data)


    def solve(self, b):
        """
        Solve system using factors computed previously.

        Parameters
        ----------
        b : ndarray
            right-hand side term

        Returns
        -------
        x: ndarray
            solution of the system
        """

        if b.dtype != np.float64:
            raise TypeError('Input data must by float64')

        if b.ndim != 1:
            raise ValueError('b must be 1D array')

        if b.size != self.row_count:
            raise ValueError('b has wrong size')

        cdef DenseVector_Double bvec
        cdef DenseVector_Double xvec

        try:
            bvec.data = <double *> malloc(self.row_count * sizeof(double))
            xvec.data = <double *> malloc(self.col_count * sizeof(double))
            bvec.count = self.row_count
            xvec.count = self.col_count

            for n in range(self.row_count):
                bvec.data[n] = b[n]

            SparseSolve(self.factors, bvec, xvec)

            x = np.empty((self.col_count,))
            for n in range(x.size):
                x[n] = xvec.data[n]
        finally:
            free(bvec.data)
            free(xvec.data)

        return x

