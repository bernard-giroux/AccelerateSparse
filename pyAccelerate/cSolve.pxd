from numpy cimport uint8_t
from libcpp cimport bool

cdef extern from "Accelerate/Accelerate.h":
    ctypedef enum SparseKind_t:
        SparseOrdinary = 0u
        SparseTriangular = 1u
        SparseUnitTriangular = 2u
        SparseSymmetric = 3u

    cpdef enum SparseTriangle_t:
        SparseUpperTriangle = 0
        SparseLowerTriangle = 1

    ctypedef enum SparseFactorization_t:
        SparseFactorizationCholesky = 0
        SparseFactorizationLDLT = 1
        SparseFactorizationLDLTUnpivoted = 2
        SparseFactorizationLDLTSBK = 3
        SparseFactorizationLDLTTPP = 4
        SparseFactorizationQR = 40
        SparseFactorizationCholeskyAtA = 41

    ctypedef enum SparseStatus_t:
        SparseStatusOK            =  0
        SparseFactorizationFailed = -1
        SparseMatrixIsSingular    = -2
        SparseInternalError       = -3
        SparseParameterError      = -4
        # SparseStatusReleased      = -INT_MAX


    ctypedef struct SparseAttributes_t:
        bool transpose
        SparseTriangle_t triangle
        SparseKind_t kind
        unsigned int _reserved
        bool _allocatedBySparse

    ctypedef struct SparseMatrixStructure:
        int rowCount
        int columnCount
        long *columnStarts
        int *rowIndices
        SparseAttributes_t attributes
        uint8_t blockSize

    ctypedef struct SparseMatrix_Double:
        SparseMatrixStructure structure
        double *data

    ctypedef struct SparseMatrix_Float:
        SparseMatrixStructure structure
        float *data
    
    ctypedef struct DenseVector_Double:
        int count
        double *data

    ctypedef struct DenseVector_Float:
        int count
        float *data

    ctypedef struct SparseOpaqueFactorization_Double:
        SparseStatus_t status

    SparseOpaqueFactorization_Double SparseFactor(SparseFactorization_t type, SparseMatrix_Double Matrix)
    void SparseRefactor(SparseMatrix_Double Matrix, SparseOpaqueFactorization_Double *Factorization)
    void SparseSolve(SparseOpaqueFactorization_Double Factored, DenseVector_Double b, DenseVector_Double x)
    void SparseCleanup(SparseOpaqueFactorization_Double factors)
    void SparseCleanup(SparseMatrix_Double matrix)
