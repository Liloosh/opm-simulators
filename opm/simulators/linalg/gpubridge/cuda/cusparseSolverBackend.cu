/*
  Copyright 2019 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#include <dune/common/timer.hh>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/simulators/linalg/gpubridge/GpuResult.hpp>
#include <opm/simulators/linalg/gpubridge/cuda/cuWellContributions.hpp>
#include <opm/simulators/linalg/gpubridge/cuda/cuda_header.hpp>
#include <opm/simulators/linalg/gpubridge/cuda/cusparseSolverBackend.hpp>

#include "cublas_v2.h"
#include "cusparse_v2.h"
// For more information about cusparse, check https://docs.nvidia.com/cuda/cusparse/index.html

// iff true, the nonzeroes of the matrix are copied row-by-row into a contiguous, pinned memory array, then a single GPU
// memcpy is done otherwise, the nonzeroes of the matrix are assumed to be in a contiguous array, and a single GPU
// memcpy is enough
#define COPY_ROW_BY_ROW 0

#include <thread>
#include <type_traits>

extern std::shared_ptr<std::thread> copyThread;

#if HAVE_OPENMP
#include <omp.h>
#endif // HAVE_OPENMP

namespace Opm::Accelerator
{

using Dune::Timer;

const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
const cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseDirection_t order = CUSPARSE_DIRECTION_ROW;

template <class Scalar>
__global__ void
computeBeta(Scalar* beta, Scalar* rho, Scalar* rhop, Scalar* alpha, Scalar* omega)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0) {
        *beta = ((*rho) / (*rhop)) * ((*alpha) / (*omega));
    }
}

template <class Scalar>
__global__ void
computeDivision(Scalar* alpha, Scalar* rho, Scalar* tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        *alpha = (*rho) / (*tmp);
    }
}

template <class Scalar>
__global__ void
makeNegative(Scalar* src, Scalar* dest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0) {
        *dest = (*src) * (-1);
    }
}

template <typename Scalar>
__global__ void
fusedVectorUpdate(
    int n, const Scalar* m_one_ptr, const Scalar* one_ptr, const Scalar* d_b, Scalar* d_r, Scalar* d_rw, Scalar* d_p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        Scalar r_val = d_r[i];
        Scalar b_val = d_b[i];

        Scalar m_one = *m_one_ptr;
        Scalar one = *one_ptr;

        r_val = (m_one * r_val) + (one * b_val);

        d_r[i] = r_val;
        d_rw[i] = r_val;
        d_p[i] = r_val;
    }
}

template <class Scalar, unsigned int block_size>
cusparseSolverBackend<Scalar, block_size>::cusparseSolverBackend(int verbosity_,
                                                                 int maxit_,
                                                                 Scalar tolerance_,
                                                                 unsigned int deviceID_,
                                                                 bool graph_enabled_,
                                                                 bool graph_viz_enabled_,
                                                                 bool fuse_vector_,
                                                                 bool graph_split_)
    : Base(verbosity_, maxit_, tolerance_, deviceID_)
    , graph_enabled(graph_enabled_)
    , graph_viz_enabled(graph_viz_enabled_)
    , fuse_vector(fuse_vector_)
    , graph_split(graph_split_)
{
    // initialize CUDA device, stream and libraries

    cudaSetDevice(deviceID);
    cudaCheckLastError("Could not get device");
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceID);
    cudaCheckLastError("Could not get device properties");
    std::ostringstream out;
    out << "Name GPU: " << props.name << ", Compute Capability: " << props.major << "." << props.minor;
    OpmLog::info(out.str());

    cudaStreamCreate(&stream);
    cudaCheckLastError("Could not create stream");

    cublasCreate(&cublasHandle);
    cudaCheckLastError("Could not create cublasHandle");
    cusparseCreate(&cusparseHandle);
    cudaCheckLastError("Could not create cusparseHandle");

    cublasSetStream(cublasHandle, stream);
    cudaCheckLastError("Could not set stream to cublas");
    cusparseSetStream(cusparseHandle, stream);
    cudaCheckLastError("Could not set stream to cusparse");

    Scalar one_graph = 1.0;
    Scalar zero_graph = 0.0;
    Scalar m_one_graph = -1.0;

    if (graph_enabled) {
        cudaMalloc((void**)&norm_0_d, sizeof(Scalar));

        cudaMalloc((void**)&m_one_graph_const_d, sizeof(Scalar));
        cudaMemcpy(m_one_graph_const_d, &m_one_graph, sizeof(Scalar), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&one_graph_const_d, sizeof(Scalar));
        cudaMemcpy(one_graph_const_d, &one_graph, sizeof(Scalar), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&rho_d, sizeof(Scalar));
        cudaMemcpy(rho_d, &one_graph, sizeof(Scalar), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&rhop_d, sizeof(Scalar));
        cudaMalloc((void**)&alpha_d, sizeof(Scalar));
        cudaMalloc((void**)&omega_d, sizeof(Scalar));
        cudaMalloc((void**)&nomega_d, sizeof(Scalar));
        cudaMalloc((void**)&beta_d, sizeof(Scalar));

        cudaMalloc((void**)&nalpha_d, sizeof(Scalar));
        cudaMalloc((void**)&tmp1_d, sizeof(Scalar));
        cudaMalloc((void**)&tmp2_d, sizeof(Scalar));
        cudaMalloc((void**)&norm_d, sizeof(Scalar));
    }
}

template <class Scalar, unsigned int block_size>
cusparseSolverBackend<Scalar, block_size>::~cusparseSolverBackend()
{
    finalize();
}

template <class Scalar, unsigned int block_size>
template <bool viz, bool fuse>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_1_create()
{
    Scalar zero = 0.0;
    Scalar one = 1.0;
    int n = N;

    cudaGraph_t graph;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    if constexpr (std::is_same_v<Scalar, float>) {
        cusparseSbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_x,
                       &zero,
                       d_r);

        if constexpr (fuse) {
            cublasSscal(cublasHandle, n, m_one_graph_const_d, d_r, 1);
            cublasSaxpy(cublasHandle, n, one_graph_const_d, d_b, 1, d_r, 1);
            cublasScopy(cublasHandle, n, d_r, 1, d_rw, 1);
            cublasScopy(cublasHandle, n, d_r, 1, d_p, 1);
        } else {
            int threadsPerBlock = 256;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            fusedVectorUpdate<Scalar><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                n, m_one_graph_const_d, one_graph_const_d, d_b, d_r, d_rw, d_p);
        }

        cublasSnrm2(cublasHandle, n, d_r, 1, norm_0_d);
    } else {
        cusparseDbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_x,
                       &zero,
                       d_r);

        if constexpr (fuse) {
            cublasDscal(cublasHandle, n, m_one_graph_const_d, d_r, 1);
            cublasDaxpy(cublasHandle, n, one_graph_const_d, d_b, 1, d_r, 1);
            cublasDcopy(cublasHandle, n, d_r, 1, d_rw, 1);
            cublasDcopy(cublasHandle, n, d_r, 1, d_p, 1);
        } else {
            int threadsPerBlock = 256;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            fusedVectorUpdate<Scalar><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                n, m_one_graph_const_d, one_graph_const_d, d_b, d_r, d_rw, d_p);
        }

        cublasDnrm2(cublasHandle, n, d_r, 1, norm_0_d);
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph1.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_1, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool viz>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_2_create()
{
    int n = N;

    cudaGraph_t graph;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    computeBeta<Scalar><<<1, 1, 0, stream>>>(beta_d, rho_d, rhop_d, alpha_d, omega_d);
    makeNegative<<<1, 1, 0, stream>>>(omega_d, nomega_d);

    if constexpr (std::is_same_v<Scalar, float>) {
        cublasSaxpy(cublasHandle, n, nomega_d, d_v, 1, d_p, 1);
        cublasSscal(cublasHandle, n, beta_d, d_p, 1);
        cublasSaxpy(cublasHandle, n, one_graph_const_d, d_r, 1, d_p, 1);
    } else {
        cublasDaxpy(cublasHandle, n, nomega_d, d_v, 1, d_p, 1);
        cublasDscal(cublasHandle, n, beta_d, d_p, 1);
        cublasDaxpy(cublasHandle, n, one_graph_const_d, d_r, 1, d_p, 1);
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph2.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_2, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool viz>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_3_create()
{
    Scalar zero = 0.0;
    Scalar one = 1.0;

    cudaGraph_t graph;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    if constexpr (std::is_same_v<Scalar, float>) {
        // apply ilu0
        cusparseSbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_L,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_L,
                              d_p,
                              d_t,
                              policy,
                              d_buffer);
        cusparseSbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_U,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_U,
                              d_t,
                              d_pw,
                              policy,
                              d_buffer);
        // spmv
        cusparseSbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_pw,
                       &zero,
                       d_v);
    } else {
        // apply ilu0
        cusparseDbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_L,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_L,
                              d_p,
                              d_t,
                              policy,
                              d_buffer);
        cusparseDbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_U,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_U,
                              d_t,
                              d_pw,
                              policy,
                              d_buffer);
        // spmv
        cusparseDbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_pw,
                       &zero,
                       d_v);
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph3.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_3, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool viz, bool split>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_4_create()
{
    int n = N;

    cudaGraph_t graph;
    cudaStream_t stream_1, stream_2;
    cudaEvent_t fork_event, join_event_1, join_event_2;

    if constexpr (split) {
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);

        cudaEventCreate(&fork_event);
        cudaEventCreate(&join_event_1);
        cudaEventCreate(&join_event_2);
    }

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if constexpr (std::is_same_v<Scalar, float>) {
        cublasSdot(cublasHandle, n, d_rw, 1, d_v, 1, tmp1_d);
    } else {
        cublasDdot(cublasHandle, n, d_rw, 1, d_v, 1, tmp1_d);
    }

    computeDivision<<<1, 1, 0, stream>>>(alpha_d, rho_d, tmp1_d);
    makeNegative<<<1, 1, 0, stream>>>(alpha_d, nalpha_d);

    if constexpr (std::is_same_v<Scalar, float>) {
        if constexpr (!split) {
            cublasSaxpy(cublasHandle, n, nalpha_d, d_v, 1, d_r, 1);
            cublasSaxpy(cublasHandle, n, alpha_d, d_pw, 1, d_x, 1);
            cublasSnrm2(cublasHandle, n, d_r, 1, norm_d);
        } else {
            cudaEventRecord(fork_event, stream);
            cudaStreamWaitEvent(stream_1, fork_event, 0);
            cudaStreamWaitEvent(stream_2, fork_event, 0);

            cublasSetStream(cublasHandle, stream_1);
            cublasSaxpy(cublasHandle, n, nalpha_d, d_v, 1, d_r, 1);
            cublasSnrm2(cublasHandle, n, d_r, 1, norm_d);

            cublasSetStream(cublasHandle, stream_2);
            cublasSaxpy(cublasHandle, n, alpha_d, d_pw, 1, d_x, 1);

            cudaEventRecord(join_event_1, stream_1);
            cudaEventRecord(join_event_2, stream_2);

            cudaStreamWaitEvent(stream, join_event_1, 0);
            cudaStreamWaitEvent(stream, join_event_2, 0);

            cublasSetStream(cublasHandle, stream);
        }
    } else {
        if constexpr (!split) {
            cublasDaxpy(cublasHandle, n, nalpha_d, d_v, 1, d_r, 1);
            cublasDaxpy(cublasHandle, n, alpha_d, d_pw, 1, d_x, 1);
            cublasDnrm2(cublasHandle, n, d_r, 1, norm_d);
        } else {
            cudaEventRecord(fork_event, stream);
            cudaStreamWaitEvent(stream_1, fork_event, 0);
            cudaStreamWaitEvent(stream_2, fork_event, 0);

            cublasSetStream(cublasHandle, stream_1);
            cublasDaxpy(cublasHandle, n, nalpha_d, d_v, 1, d_r, 1);
            cublasDnrm2(cublasHandle, n, d_r, 1, norm_d);

            cublasSetStream(cublasHandle, stream_2);
            cublasDaxpy(cublasHandle, n, alpha_d, d_pw, 1, d_x, 1);

            cudaEventRecord(join_event_1, stream_1);
            cudaEventRecord(join_event_2, stream_2);

            cudaStreamWaitEvent(stream, join_event_1, 0);
            cudaStreamWaitEvent(stream, join_event_2, 0);

            cublasSetStream(cublasHandle, stream);
        }
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (split) {
        cudaEventDestroy(fork_event);
        cudaEventDestroy(join_event_1);
        cudaEventDestroy(join_event_2);

        cudaStreamDestroy(stream_1);
        cudaStreamDestroy(stream_2);
    }

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph4.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_4, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool viz>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_5_create()
{
    Scalar zero = 0.0;
    Scalar one = 1.0;

    cudaGraph_t graph;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    if constexpr (std::is_same_v<Scalar, float>) {
        // apply ilu0
        cusparseSbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_L,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_L,
                              d_r,
                              d_t,
                              policy,
                              d_buffer);

        cusparseSbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_U,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_U,
                              d_t,
                              d_s,
                              policy,
                              d_buffer);

        // spmv
        cusparseSbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_s,
                       &zero,
                       d_t);
    } else {
        // apply ilu0
        cusparseDbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_L,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_L,
                              d_r,
                              d_t,
                              policy,
                              d_buffer);

        cusparseDbsrsv2_solve(cusparseHandle,
                              order,
                              operation,
                              Nb,
                              nnzbs_prec,
                              &one,
                              descr_U,
                              d_mVals,
                              d_mRows,
                              d_mCols,
                              block_size,
                              info_U,
                              d_t,
                              d_s,
                              policy,
                              d_buffer);

        // spmv
        cusparseDbsrmv(cusparseHandle,
                       order,
                       operation,
                       Nb,
                       Nb,
                       nnzb,
                       &one,
                       descr_M,
                       d_bVals,
                       d_bRows,
                       d_bCols,
                       block_size,
                       d_s,
                       &zero,
                       d_t);
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph5.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_5, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool viz, bool split>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab_graph_6_create()
{
    int n = N;

    cudaGraph_t graph;
    cudaStream_t stream_1, stream_2;
    cudaEvent_t fork_event, join_event_1, join_event_2;

    if constexpr (split) {
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);

        cudaEventCreate(&fork_event);
        cudaEventCreate(&join_event_1);
        cudaEventCreate(&join_event_2);
    }


    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    if constexpr (std::is_same_v<Scalar, float>) {
        cublasSdot(cublasHandle, n, d_t, 1, d_r, 1, tmp1_d);
        cublasSdot(cublasHandle, n, d_t, 1, d_t, 1, tmp2_d);
    } else {
        cublasDdot(cublasHandle, n, d_t, 1, d_r, 1, tmp1_d);
        cublasDdot(cublasHandle, n, d_t, 1, d_t, 1, tmp2_d);
    }

    computeDivision<<<1, 1, 0, stream>>>(omega_d, tmp1_d, tmp2_d);
    makeNegative<<<1, 1, 0, stream>>>(omega_d, nomega_d);

    if constexpr (std::is_same_v<Scalar, float>) {
        if constexpr (!split) {
            cublasSaxpy(cublasHandle, n, omega_d, d_s, 1, d_x, 1);
            cublasSaxpy(cublasHandle, n, nomega_d, d_t, 1, d_r, 1);
            cublasSnrm2(cublasHandle, n, d_r, 1, norm_d);
        } else {
            cudaEventRecord(fork_event, stream);
            cudaStreamWaitEvent(stream_1, fork_event, 0);
            cudaStreamWaitEvent(stream_2, fork_event, 0);

            cublasSetStream(cublasHandle, stream_1);
            cublasSaxpy(cublasHandle, n, nomega_d, d_t, 1, d_r, 1);
            cublasSnrm2(cublasHandle, n, d_r, 1, norm_d);

            cublasSetStream(cublasHandle, stream_2);
            cublasSaxpy(cublasHandle, n, omega_d, d_s, 1, d_x, 1);

            cudaEventRecord(join_event_1, stream_1);
            cudaEventRecord(join_event_2, stream_2);

            cudaStreamWaitEvent(stream, join_event_1, 0);
            cudaStreamWaitEvent(stream, join_event_2, 0);

            cublasSetStream(cublasHandle, stream);
        }
    } else {
        if constexpr (!split) {
            cublasDaxpy(cublasHandle, n, omega_d, d_s, 1, d_x, 1);
            cublasDaxpy(cublasHandle, n, nomega_d, d_t, 1, d_r, 1);
            cublasDnrm2(cublasHandle, n, d_r, 1, norm_d);
        } else {
            cudaEventRecord(fork_event, stream);
            cudaStreamWaitEvent(stream_1, fork_event, 0);
            cudaStreamWaitEvent(stream_2, fork_event, 0);

            cublasSetStream(cublasHandle, stream_1);
            cublasDaxpy(cublasHandle, n, nomega_d, d_t, 1, d_r, 1);
            cublasDnrm2(cublasHandle, n, d_r, 1, norm_d);

            cublasSetStream(cublasHandle, stream_2);
            cublasDaxpy(cublasHandle, n, omega_d, d_s, 1, d_x, 1);

            cudaEventRecord(join_event_1, stream_1);
            cudaEventRecord(join_event_2, stream_2);

            cudaStreamWaitEvent(stream, join_event_1, 0);
            cudaStreamWaitEvent(stream, join_event_2, 0);

            cublasSetStream(cublasHandle, stream);
        }
    }

    cudaStreamEndCapture(stream, &graph);

    if constexpr (split) {
        cudaEventDestroy(fork_event);
        cudaEventDestroy(join_event_1);
        cudaEventDestroy(join_event_2);

        cudaStreamDestroy(stream_1);
        cudaStreamDestroy(stream_2);
    }

    if constexpr (viz) {
        cudaGraphDebugDotPrint(graph, "graph6.dot", cudaGraphDebugDotFlagsVerbose);
    }

    cudaGraphInstantiate(&graphExec_6, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
}

template <class Scalar, unsigned int block_size>
template <bool enabled, bool viz, bool fuse, bool split>
void
cusparseSolverBackend<Scalar, block_size>::gpu_pbicgstab(WellContributions<Scalar>& wellContribs, GpuResult& res)
{
    Timer t_total, t_prec(false), t_spmv(false), t_well(false), t_rest(false);
    int n = N;
    Scalar rho = 1.0, rhop;
    Scalar alpha, nalpha, beta;
    Scalar omega, nomega, tmp1, tmp2;
    Scalar norm, norm_0;
    Scalar zero = 0.0;
    Scalar one = 1.0;
    Scalar mone = -1.0;
    float it;

    if constexpr (enabled) {
        cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    }

    if (wellContribs.getNumWells() > 0) {
        static_cast<WellContributionsCuda<Scalar>&>(wellContribs).setCudaStream(stream);
    }

    if constexpr (!enabled) {
        if constexpr (std::is_same_v<Scalar, float>) {
            cusparseSbsrmv(cusparseHandle,
                           order,
                           operation,
                           Nb,
                           Nb,
                           nnzb,
                           &one,
                           descr_M,
                           d_bVals,
                           d_bRows,
                           d_bCols,
                           block_size,
                           d_x,
                           &zero,
                           d_r);
        } else {
            cusparseDbsrmv(cusparseHandle,
                           order,
                           operation,
                           Nb,
                           Nb,
                           nnzb,
                           &one,
                           descr_M,
                           d_bVals,
                           d_bRows,
                           d_bCols,
                           block_size,
                           d_x,
                           &zero,
                           d_r);
        }

        if constexpr (std::is_same_v<Scalar, float>) {
            cublasSscal(cublasHandle, n, &mone, d_r, 1);
            cublasSaxpy(cublasHandle, n, &one, d_b, 1, d_r, 1);
            cublasScopy(cublasHandle, n, d_r, 1, d_rw, 1);
            cublasScopy(cublasHandle, n, d_r, 1, d_p, 1);
            cublasSnrm2(cublasHandle, n, d_r, 1, &norm_0);
        } else {
            cublasDscal(cublasHandle, n, &mone, d_r, 1);
            cublasDaxpy(cublasHandle, n, &one, d_b, 1, d_r, 1);
            cublasDcopy(cublasHandle, n, d_r, 1, d_rw, 1);
            cublasDcopy(cublasHandle, n, d_r, 1, d_p, 1);
            cublasDnrm2(cublasHandle, n, d_r, 1, &norm_0);
        }
    } else {
        if (!isCaptured_1) {
            gpu_pbicgstab_graph_1_create<viz, fuse>();
            isCaptured_1 = true;
        }
        cudaGraphLaunch(graphExec_1, stream);
        cudaMemcpy(&norm_0, norm_0_d, sizeof(Scalar), cudaMemcpyDeviceToHost);
    }

    if (verbosity > 1) {
        std::ostringstream out;
        out << std::scientific << "cusparseSolver initial norm: " << norm_0;
        OpmLog::info(out.str());
    }

    for (it = 0.5; it < maxit; it += 0.5) {
        if constexpr (!enabled) {
            rhop = rho;

            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSdot(cublasHandle, n, d_rw, 1, d_r, 1, &rho);
            } else {
                cublasDdot(cublasHandle, n, d_rw, 1, d_r, 1, &rho);
            }
        } else {
            cudaMemcpy(rhop_d, rho_d, sizeof(Scalar), cudaMemcpyDeviceToDevice);

            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSdot(cublasHandle, n, d_rw, 1, d_r, 1, rho_d);
            } else {
                cublasDdot(cublasHandle, n, d_rw, 1, d_r, 1, rho_d);
            }
        }

        if (it > 1) {
            if constexpr (!enabled) {
                beta = (rho / rhop) * (alpha / omega);
                nomega = -omega;

                if constexpr (std::is_same_v<Scalar, float>) {
                    cublasSaxpy(cublasHandle, n, &nomega, d_v, 1, d_p, 1);
                    cublasSscal(cublasHandle, n, &beta, d_p, 1);
                    cublasSaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1);
                } else {
                    cublasDaxpy(cublasHandle, n, &nomega, d_v, 1, d_p, 1);
                    cublasDscal(cublasHandle, n, &beta, d_p, 1);
                    cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1);
                }
            } else {
                if (!isCaptured_2) {
                    gpu_pbicgstab_graph_2_create<viz>();
                    isCaptured_2 = true;
                }
                cudaGraphLaunch(graphExec_2, stream);
            }
        }

        if constexpr (!enabled) {
            if constexpr (std::is_same_v<Scalar, float>) {
                // apply ilu0
                cusparseSbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_L,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_L,
                                      d_p,
                                      d_t,
                                      policy,
                                      d_buffer);
                cusparseSbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_U,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_U,
                                      d_t,
                                      d_pw,
                                      policy,
                                      d_buffer);
                // spmv
                cusparseSbsrmv(cusparseHandle,
                               order,
                               operation,
                               Nb,
                               Nb,
                               nnzb,
                               &one,
                               descr_M,
                               d_bVals,
                               d_bRows,
                               d_bCols,
                               block_size,
                               d_pw,
                               &zero,
                               d_v);
            } else {
                // apply ilu0
                cusparseDbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_L,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_L,
                                      d_p,
                                      d_t,
                                      policy,
                                      d_buffer);
                cusparseDbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_U,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_U,
                                      d_t,
                                      d_pw,
                                      policy,
                                      d_buffer);
                // spmv
                cusparseDbsrmv(cusparseHandle,
                               order,
                               operation,
                               Nb,
                               Nb,
                               nnzb,
                               &one,
                               descr_M,
                               d_bVals,
                               d_bRows,
                               d_bCols,
                               block_size,
                               d_pw,
                               &zero,
                               d_v);
            }
        } else {
            if (!isCaptured_3) {
                gpu_pbicgstab_graph_3_create<viz>();
                isCaptured_3 = true;
            }
            cudaGraphLaunch(graphExec_3, stream);
        }

        // apply wellContributions
        if (wellContribs.getNumWells() > 0) {
            static_cast<WellContributionsCuda<Scalar>&>(wellContribs).apply(d_pw, d_v);
        }

        if constexpr (!enabled) {
            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSdot(cublasHandle, n, d_rw, 1, d_v, 1, &tmp1);
            } else {
                cublasDdot(cublasHandle, n, d_rw, 1, d_v, 1, &tmp1);
            }

            alpha = rho / tmp1;
            nalpha = -alpha;

            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSaxpy(cublasHandle, n, &nalpha, d_v, 1, d_r, 1);
                cublasSaxpy(cublasHandle, n, &alpha, d_pw, 1, d_x, 1);
                cublasSnrm2(cublasHandle, n, d_r, 1, &norm);
            } else {
                cublasDaxpy(cublasHandle, n, &nalpha, d_v, 1, d_r, 1);
                cublasDaxpy(cublasHandle, n, &alpha, d_pw, 1, d_x, 1);
                cublasDnrm2(cublasHandle, n, d_r, 1, &norm);
            }
        } else {
            if (!isCaptured_4) {
                gpu_pbicgstab_graph_4_create<viz, split>();
                isCaptured_4 = true;
            }
            cudaGraphLaunch(graphExec_4, stream);
            cudaMemcpy(&norm, norm_d, sizeof(Scalar), cudaMemcpyDeviceToHost);
        }

        if (norm < tolerance * norm_0) {
            break;
        }

        it += 0.5;

        if constexpr (!enabled) {
            if constexpr (std::is_same_v<Scalar, float>) {
                // apply ilu0
                cusparseSbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_L,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_L,
                                      d_r,
                                      d_t,
                                      policy,
                                      d_buffer);

                cusparseSbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_U,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_U,
                                      d_t,
                                      d_s,
                                      policy,
                                      d_buffer);

                // spmv
                cusparseSbsrmv(cusparseHandle,
                               order,
                               operation,
                               Nb,
                               Nb,
                               nnzb,
                               &one,
                               descr_M,
                               d_bVals,
                               d_bRows,
                               d_bCols,
                               block_size,
                               d_s,
                               &zero,
                               d_t);
            } else {
                // apply ilu0
                cusparseDbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_L,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_L,
                                      d_r,
                                      d_t,
                                      policy,
                                      d_buffer);

                cusparseDbsrsv2_solve(cusparseHandle,
                                      order,
                                      operation,
                                      Nb,
                                      nnzbs_prec,
                                      &one,
                                      descr_U,
                                      d_mVals,
                                      d_mRows,
                                      d_mCols,
                                      block_size,
                                      info_U,
                                      d_t,
                                      d_s,
                                      policy,
                                      d_buffer);

                // spmv
                cusparseDbsrmv(cusparseHandle,
                               order,
                               operation,
                               Nb,
                               Nb,
                               nnzb,
                               &one,
                               descr_M,
                               d_bVals,
                               d_bRows,
                               d_bCols,
                               block_size,
                               d_s,
                               &zero,
                               d_t);
            }
        } else {
            if (!isCaptured_5) {
                gpu_pbicgstab_graph_5_create<viz>();
                isCaptured_5 = true;
            }
            cudaGraphLaunch(graphExec_5, stream);
            cudaMemcpy(&norm, norm_d, sizeof(Scalar), cudaMemcpyDeviceToHost);
        }

        // apply wellContributions
        if (wellContribs.getNumWells() > 0) {
            static_cast<WellContributionsCuda<Scalar>&>(wellContribs).apply(d_s, d_t);
        }

        if constexpr (!enabled) {
            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSdot(cublasHandle, n, d_t, 1, d_r, 1, &tmp1);
                cublasSdot(cublasHandle, n, d_t, 1, d_t, 1, &tmp2);
            } else {
                cublasDdot(cublasHandle, n, d_t, 1, d_r, 1, &tmp1);
                cublasDdot(cublasHandle, n, d_t, 1, d_t, 1, &tmp2);
            }

            omega = tmp1 / tmp2;
            nomega = -omega;

            if constexpr (std::is_same_v<Scalar, float>) {
                cublasSaxpy(cublasHandle, n, &omega, d_s, 1, d_x, 1);
                cublasSaxpy(cublasHandle, n, &nomega, d_t, 1, d_r, 1);
                cublasSnrm2(cublasHandle, n, d_r, 1, &norm);
            } else {
                cublasDaxpy(cublasHandle, n, &omega, d_s, 1, d_x, 1);
                cublasDaxpy(cublasHandle, n, &nomega, d_t, 1, d_r, 1);
                cublasDnrm2(cublasHandle, n, d_r, 1, &norm);
            }
        } else {
            if (!isCaptured_6) {
                gpu_pbicgstab_graph_6_create<viz, split>();
                isCaptured_6 = true;
            }

            cudaGraphLaunch(graphExec_6, stream);
            cudaMemcpy(&norm, norm_d, sizeof(Scalar), cudaMemcpyDeviceToHost);
        }

        if (norm < tolerance * norm_0) {
            break;
        }

        if (verbosity > 1) {
            std::ostringstream out;
            out << "it: " << it << std::scientific << ", norm: " << norm;
            OpmLog::info(out.str());
        }
    }

    if constexpr (enabled) {
        cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);
    }

    res.iterations = std::min(it, (float)maxit);
    res.reduction = norm / norm_0;
    res.conv_rate = static_cast<double>(pow(res.reduction, 1.0 / it));
    res.elapsed = t_total.stop();
    res.converged = (it != (maxit + 0.5));

    if (verbosity > 0) {
        std::ostringstream out;
        out << "=== converged: " << res.converged << ", conv_rate: " << res.conv_rate << ", time: " << res.elapsed
            << ", time per iteration: " << res.elapsed / it << ", iterations: " << it;
        OpmLog::info(out.str());
    }
}

template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::initialize(std::shared_ptr<BlockedMatrix<Scalar>> matrix,
                                                      std::shared_ptr<BlockedMatrix<Scalar>> jacMatrix)
{
    this->Nb = matrix->Nb;
    this->N = Nb * block_size;
    this->nnzb = matrix->nnzbs;
    this->nnz = nnzb * block_size * block_size;

    if (jacMatrix) {
        useJacMatrix = true;
        nnzbs_prec = jacMatrix->nnzbs;
    } else {
        nnzbs_prec = nnzb;
    }

    std::ostringstream out;
    out << "Initializing GPU, matrix size: " << Nb << " blockrows, nnz: " << nnzb << " blocks\n";
    if (useJacMatrix) {
        out << "Blocks in ILU matrix: " << nnzbs_prec << "\n";
    }
    out << "Maxit: " << maxit << std::scientific << ", tolerance: " << tolerance << "\n";
    OpmLog::info(out.str());

    cudaMalloc((void**)&d_x, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_b, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_r, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_rw, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_p, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_pw, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_s, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_t, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_v, sizeof(Scalar) * N);
    cudaMalloc((void**)&d_bVals, sizeof(Scalar) * nnz);
    cudaMalloc((void**)&d_bCols, sizeof(int) * nnzb);
    cudaMalloc((void**)&d_bRows, sizeof(int) * (Nb + 1));
    if (useJacMatrix) {
        cudaMalloc((void**)&d_mVals, sizeof(Scalar) * nnzbs_prec * block_size * block_size);
        cudaMalloc((void**)&d_mCols, sizeof(int) * nnzbs_prec);
        cudaMalloc((void**)&d_mRows, sizeof(int) * (Nb + 1));
    } else {
        cudaMalloc((void**)&d_mVals, sizeof(Scalar) * nnz);
        d_mCols = d_bCols;
        d_mRows = d_bRows;
    }
    cudaCheckLastError("Could not allocate enough memory on GPU");

#if COPY_ROW_BY_ROW
    cudaMallocHost((void**)&vals_contiguous, sizeof(Scalar) * nnz);
    cudaCheckLastError("Could not allocate pinned memory");
#endif

    initialized = true;
} // end initialize()

template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::finalize()
{
    if (initialized) {
        cudaFree(d_x);
        cudaFree(d_b);
        cudaFree(d_r);
        cudaFree(d_rw);
        cudaFree(d_p);
        cudaFree(d_pw);
        cudaFree(d_s);
        cudaFree(d_t);
        cudaFree(d_v);
        cudaFree(d_mVals);
        if (useJacMatrix) {
            cudaFree(d_mCols);
            cudaFree(d_mRows);
        }

        if (graph_enabled) {
            cudaFree(norm_0_d);
            cudaFree(norm_d);
            cudaFree(m_one_graph_const_d);
            cudaFree(one_graph_const_d);

            cudaFree(beta_d);
            cudaFree(omega_d);
            cudaFree(nomega_d);
            cudaFree(alpha_d);
            cudaFree(nalpha_d);
            cudaFree(rho_d);
            cudaFree(rhop_d);

            cudaFree(tmp1_d);
            cudaFree(tmp2_d);

            cudaGraphExecDestroy(graphExec_1);
            cudaGraphExecDestroy(graphExec_2);
            cudaGraphExecDestroy(graphExec_3);
            cudaGraphExecDestroy(graphExec_4);
            cudaGraphExecDestroy(graphExec_5);
            cudaGraphExecDestroy(graphExec_6);
        }

        cudaFree(d_bVals);
        cudaFree(d_bCols);
        cudaFree(d_bRows);
        cudaFree(d_buffer);
        cusparseDestroyBsrilu02Info(info_M);
        cusparseDestroyBsrsv2Info(info_L);
        cusparseDestroyBsrsv2Info(info_U);
        cusparseDestroyMatDescr(descr_B);
        cusparseDestroyMatDescr(descr_M);
        cusparseDestroyMatDescr(descr_L);
        cusparseDestroyMatDescr(descr_U);
        cusparseDestroy(cusparseHandle);
        cublasDestroy(cublasHandle);
#if COPY_ROW_BY_ROW
        cudaFreeHost(vals_contiguous);
#endif
        cudaStreamDestroy(stream);
    }
} // end finalize()

template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::copy_system_to_gpu(std::shared_ptr<BlockedMatrix<Scalar>> matrix,
                                                              Scalar* b,
                                                              std::shared_ptr<BlockedMatrix<Scalar>> jacMatrix)
{
    Timer t;

    cudaMemcpyAsync(d_bCols, matrix->colIndices, nnzb * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bRows, matrix->rowPointers, (Nb + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b, N * sizeof(Scalar), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_x, 0, N * sizeof(Scalar), stream);

#if COPY_ROW_BY_ROW
    int sum = 0;
    for (int i = 0; i < Nb; ++i) {
        int size_row = matrix->rowPointers[i + 1] - matrix->rowPointers[i];
        memcpy(vals_contiguous + sum, matrix->nnzValues + sum, size_row * sizeof(Scalar) * block_size * block_size);
        sum += size_row * block_size * block_size;
    }
    cudaMemcpyAsync(d_bVals, vals_contiguous, nnz * sizeof(Scalar), cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpyAsync(d_bVals, matrix->nnzValues, nnz * sizeof(Scalar), cudaMemcpyHostToDevice, stream);

    bool use_multithreading = true;
#if HAVE_OPENMP
    if (omp_get_max_threads() == 1)
        use_multithreading = false;
#endif

    if (useJacMatrix) {
        if (use_multithreading)
            copyThread->join();

        cudaMemcpyAsync(d_mVals,
                        jacMatrix->nnzValues,
                        nnzbs_prec * block_size * block_size * sizeof(Scalar),
                        cudaMemcpyHostToDevice,
                        stream);
    } else {
        cudaMemcpyAsync(d_mVals, d_bVals, nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream);
    }
#endif

    if (useJacMatrix) {
        cudaMemcpyAsync(d_mCols, jacMatrix->colIndices, nnzbs_prec * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mRows, jacMatrix->rowPointers, (Nb + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    if (verbosity >= 3) {
        cudaStreamSynchronize(stream);

        c_copy += t.stop();
        std::ostringstream out;
        out << "---cusparseSolver::copy_system_to_gpu(): " << t.elapsed() << " s";
        OpmLog::info(out.str());
    }
} // end copy_system_to_gpu()

// don't copy rowpointers and colindices, they stay the same
template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::update_system_on_gpu(std::shared_ptr<BlockedMatrix<Scalar>> matrix,
                                                                Scalar* b,
                                                                std::shared_ptr<BlockedMatrix<Scalar>> jacMatrix)
{
    Timer t;

    cudaMemcpyAsync(d_b, b, N * sizeof(Scalar), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_x, 0, sizeof(Scalar) * N, stream);

#if COPY_ROW_BY_ROW
    int sum = 0;
    for (int i = 0; i < Nb; ++i) {
        int size_row = matrix->rowPointers[i + 1] - matrix->rowPointers[i];
        memcpy(vals_contiguous + sum, matrix->nnzValues + sum, size_row * sizeof(Scalar) * block_size * block_size);
        sum += size_row * block_size * block_size;
    }
    cudaMemcpyAsync(d_bVals, vals_contiguous, nnz * sizeof(Scalar), cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpyAsync(d_bVals, matrix->nnzValues, nnz * sizeof(Scalar), cudaMemcpyHostToDevice, stream);

    bool use_multithreading = true;
#if HAVE_OPENMP
    if (omp_get_max_threads() == 1)
        use_multithreading = false;
#endif

    if (useJacMatrix) {
        if (use_multithreading)
            copyThread->join();

        cudaMemcpyAsync(d_mVals,
                        jacMatrix->nnzValues,
                        nnzbs_prec * block_size * block_size * sizeof(Scalar),
                        cudaMemcpyHostToDevice,
                        stream);
    } else {
        cudaMemcpyAsync(d_mVals, d_bVals, nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream);
    }
#endif

    if (verbosity >= 3) {
        cudaStreamSynchronize(stream);

        c_copy += t.stop();
        std::ostringstream out;
        out << "-----cusparseSolver::update_system_on_gpu(): " << t.elapsed() << " s\n";
        out << "---cusparseSolver::cum copy: " << c_copy << " s";
        OpmLog::info(out.str());
    }
} // end update_system_on_gpu()

template <class Scalar, unsigned int block_size>
bool
cusparseSolverBackend<Scalar, block_size>::analyse_matrix()
{
    int d_bufferSize_M, d_bufferSize_L, d_bufferSize_U, d_bufferSize;
    Timer t;

    cusparseCreateMatDescr(&descr_B);
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatType(descr_B, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
    const cusparseIndexBase_t base_type = CUSPARSE_INDEX_BASE_ZERO; // matrices from Flow are base0

    cusparseSetMatIndexBase(descr_B, base_type);
    cusparseSetMatIndexBase(descr_M, base_type);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, base_type);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, base_type);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cudaCheckLastError("Could not initialize matrix descriptions");

    cusparseCreateBsrilu02Info(&info_M);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);
    cudaCheckLastError("Could not create analysis info");

    if constexpr (std::is_same_v<Scalar, float>) {
        cusparseSbsrilu02_bufferSize(cusparseHandle,
                                     order,
                                     Nb,
                                     nnzbs_prec,
                                     descr_M,
                                     d_mVals,
                                     d_mRows,
                                     d_mCols,
                                     block_size,
                                     info_M,
                                     &d_bufferSize_M);
        cusparseSbsrsv2_bufferSize(cusparseHandle,
                                   order,
                                   operation,
                                   Nb,
                                   nnzbs_prec,
                                   descr_L,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_L,
                                   &d_bufferSize_L);
        cusparseSbsrsv2_bufferSize(cusparseHandle,
                                   order,
                                   operation,
                                   Nb,
                                   nnzbs_prec,
                                   descr_U,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_U,
                                   &d_bufferSize_U);
    } else {
        cusparseDbsrilu02_bufferSize(cusparseHandle,
                                     order,
                                     Nb,
                                     nnzbs_prec,
                                     descr_M,
                                     d_mVals,
                                     d_mRows,
                                     d_mCols,
                                     block_size,
                                     info_M,
                                     &d_bufferSize_M);
        cusparseDbsrsv2_bufferSize(cusparseHandle,
                                   order,
                                   operation,
                                   Nb,
                                   nnzbs_prec,
                                   descr_L,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_L,
                                   &d_bufferSize_L);
        cusparseDbsrsv2_bufferSize(cusparseHandle,
                                   order,
                                   operation,
                                   Nb,
                                   nnzbs_prec,
                                   descr_U,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_U,
                                   &d_bufferSize_U);
    }

    d_bufferSize = std::max(d_bufferSize_M, std::max(d_bufferSize_L, d_bufferSize_U));

    cudaMalloc((void**)&d_buffer, d_bufferSize);

    // analysis of ilu LU decomposition
    if constexpr (std::is_same_v<Scalar, float>) {
        cusparseSbsrilu02_analysis(cusparseHandle,
                                   order,
                                   Nb,
                                   nnzbs_prec,
                                   descr_B,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_M,
                                   policy,
                                   d_buffer);
    } else {
        cusparseDbsrilu02_analysis(cusparseHandle,
                                   order,
                                   Nb,
                                   nnzbs_prec,
                                   descr_B,
                                   d_mVals,
                                   d_mRows,
                                   d_mCols,
                                   block_size,
                                   info_M,
                                   policy,
                                   d_buffer);
    }

    int structural_zero;
    cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        return false;
    }

    // analysis of ilu apply
    if constexpr (std::is_same_v<Scalar, float>) {
        cusparseSbsrsv2_analysis(cusparseHandle,
                                 order,
                                 operation,
                                 Nb,
                                 nnzbs_prec,
                                 descr_L,
                                 d_mVals,
                                 d_mRows,
                                 d_mCols,
                                 block_size,
                                 info_L,
                                 policy,
                                 d_buffer);
        cusparseSbsrsv2_analysis(cusparseHandle,
                                 order,
                                 operation,
                                 Nb,
                                 nnzbs_prec,
                                 descr_U,
                                 d_mVals,
                                 d_mRows,
                                 d_mCols,
                                 block_size,
                                 info_U,
                                 policy,
                                 d_buffer);
    } else {
        cusparseDbsrsv2_analysis(cusparseHandle,
                                 order,
                                 operation,
                                 Nb,
                                 nnzbs_prec,
                                 descr_L,
                                 d_mVals,
                                 d_mRows,
                                 d_mCols,
                                 block_size,
                                 info_L,
                                 policy,
                                 d_buffer);
        cusparseDbsrsv2_analysis(cusparseHandle,
                                 order,
                                 operation,
                                 Nb,
                                 nnzbs_prec,
                                 descr_U,
                                 d_mVals,
                                 d_mRows,
                                 d_mCols,
                                 block_size,
                                 info_U,
                                 policy,
                                 d_buffer);
    }
    cudaCheckLastError("Could not analyse level information");

    if (verbosity > 2) {
        cudaStreamSynchronize(stream);
        std::ostringstream out;
        out << "cusparseSolver::analyse_matrix(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }

    analysis_done = true;

    return true;
} // end analyse_matrix()

template <class Scalar, unsigned int block_size>
bool
cusparseSolverBackend<Scalar, block_size>::create_preconditioner()
{
    Timer t;

    if constexpr (std::is_same_v<Scalar, float>) {
        cusparseSbsrilu02(cusparseHandle,
                          order,
                          Nb,
                          nnzbs_prec,
                          descr_M,
                          d_mVals,
                          d_mRows,
                          d_mCols,
                          block_size,
                          info_M,
                          policy,
                          d_buffer);
    } else {
        cusparseDbsrilu02(cusparseHandle,
                          order,
                          Nb,
                          nnzbs_prec,
                          descr_M,
                          d_mVals,
                          d_mRows,
                          d_mCols,
                          block_size,
                          info_M,
                          policy,
                          d_buffer);
    }
    cudaCheckLastError("Could not perform ilu decomposition");

    int structural_zero;
    // cusparseXbsrilu02_zeroPivot() calls cudaDeviceSynchronize()
    cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(cusparseHandle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        return false;
    }

    if (verbosity > 2) {
        cudaStreamSynchronize(stream);
        std::ostringstream out;
        out << "cusparseSolver::create_preconditioner(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }
    return true;
} // end create_preconditioner()

template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::solve_system(WellContributions<Scalar>& wellContribs, GpuResult& res)
{
    // actually solve
    if (graph_enabled) {
        if (graph_viz_enabled && fuse_vector && graph_split) {
            gpu_pbicgstab<true, true, true, true>(wellContribs, res);
        } else if (graph_viz_enabled && fuse_vector && !graph_split) {
            gpu_pbicgstab<true, true, true, false>(wellContribs, res);
        } else if (graph_viz_enabled && !fuse_vector && graph_split) {
            gpu_pbicgstab<true, true, false, true>(wellContribs, res);
        } else if (graph_viz_enabled && !fuse_vector && !graph_split) {
            gpu_pbicgstab<true, true, false, false>(wellContribs, res);
        } else if (!graph_viz_enabled && fuse_vector && graph_split) {
            gpu_pbicgstab<true, false, true, true>(wellContribs, res);
        } else if (!graph_viz_enabled && fuse_vector && !graph_split) {
            gpu_pbicgstab<true, false, true, false>(wellContribs, res);
        } else if (!graph_viz_enabled && !fuse_vector && graph_split) {
            gpu_pbicgstab<true, false, false, true>(wellContribs, res);
        } else {
            gpu_pbicgstab<true, false, false, false>(wellContribs, res);
        }
    } else {
        gpu_pbicgstab<false>(wellContribs, res);
    }

    cudaStreamSynchronize(stream);
    cudaCheckLastError("Something went wrong during the GPU solve");
} // end solve_system()

// copy result to host memory
// caller must be sure that x is a valid array
template <class Scalar, unsigned int block_size>
void
cusparseSolverBackend<Scalar, block_size>::get_result(Scalar* x)
{
    Timer t;

    cudaMemcpyAsync(x, d_x, N * sizeof(Scalar), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (verbosity > 2) {
        std::ostringstream out;
        out << "cusparseSolver::get_result(): " << t.stop() << " s";
        OpmLog::info(out.str());
    }
} // end get_result()

template <class Scalar, unsigned int block_size>
SolverStatus
cusparseSolverBackend<Scalar, block_size>::solve_system(std::shared_ptr<BlockedMatrix<Scalar>> matrix,
                                                        Scalar* b,
                                                        std::shared_ptr<BlockedMatrix<Scalar>> jacMatrix,
                                                        WellContributions<Scalar>& wellContribs,
                                                        GpuResult& res)
{
    if (initialized == false) {
        initialize(matrix, jacMatrix);
        copy_system_to_gpu(matrix, b, jacMatrix);
    } else {
        update_system_on_gpu(matrix, b, jacMatrix);
    }
    if (analysis_done == false) {
        if (!analyse_matrix()) {
            return SolverStatus::GPU_SOLVER_ANALYSIS_FAILED;
        }
    }
    if (create_preconditioner()) {
        solve_system(wellContribs, res);
    } else {
        return SolverStatus::GPU_SOLVER_CREATE_PRECONDITIONER_FAILED;
    }
    return SolverStatus::GPU_SOLVER_SUCCESS;
}

#define INSTANTIATE_TYPE(T)                                                                                            \
    template class cusparseSolverBackend<T, 1>;                                                                        \
    template class cusparseSolverBackend<T, 2>;                                                                        \
    template class cusparseSolverBackend<T, 3>;                                                                        \
    template class cusparseSolverBackend<T, 4>;                                                                        \
    template class cusparseSolverBackend<T, 5>;                                                                        \
    template class cusparseSolverBackend<T, 6>;

INSTANTIATE_TYPE(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_TYPE(float)
#endif

} // namespace Opm::Accelerator
