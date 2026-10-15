/*
  Copyright 2016 IRIS AS
  Copyright 2019, 2020 Equinor ASA
  Copyright 2020 SINTEF Digital, Mathematics and Cybernetics

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
#include <opm/common/TimingMacros.hpp>
#include <opm/simulators/linalg/ISTLSolver.hpp>

#include <dune/istl/schwarz.hh>

#include <opm/grid/CpGrid.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/ParallelIstlInformation.hpp>
#include <opm/simulators/utils/ParallelCommunication.hpp>

#include <fmt/format.h>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/simulators/utils/gatherDeferredLogger.hpp>
#include <algorithm>
#include <vector>
#include <string>

#if COMPILE_GPU_BRIDGE
#include <opm/simulators/linalg/gpubridge/GpuBridge.hpp>
#include <opm/simulators/linalg/gpubridge/WellContributions.hpp>
#endif

namespace Opm {
namespace detail {

#ifdef HAVE_MPI
void copyParValues(std::any& parallelInformation, std::size_t size,
                   Dune::OwnerOverlapCopyCommunication<int,int>& comm)
{
  if (parallelInformation.type() == typeid(ParallelISTLInformation)) {
      const ParallelISTLInformation* parinfo = std::any_cast<ParallelISTLInformation>(&parallelInformation);
      assert(parinfo);
      parinfo->copyValuesTo(comm.indexSet(), comm.remoteIndices(), size, 1);
  }
}
#endif

template<class Matrix>
void makeOverlapRowsInvalid(Matrix& matrix,
                            const std::vector<int>& overlapRows)
{
    //value to set on diagonal
    const int numEq = Matrix::block_type::rows;
    typename Matrix::block_type diag_block(0.0);
    for (int eq = 0; eq < numEq; ++eq)
        diag_block[eq][eq] = 1.0;

    //loop over precalculated overlap rows and columns
    for (const auto row : overlapRows)
    {
        // Zero out row.
        matrix[row] = 0.0;

        //diagonal block set to diag(1.0).
        matrix[row][row] = diag_block;
    }
}

#if HAVE_MPI
template<class Matrix, class Comm>
void logMatrixPartitioning(const Matrix& matrix, const Comm* comm, std::size_t interiorCellNum)
{
    if (!comm) return;
    auto basic_comm = comm->communicator();
    if (basic_comm.size() <= 1) return;

    static bool already_logged = false;
    if (already_logged) return;
    already_logged = true;

    const int rank = basic_comm.rank();
    const int mpiSize = basic_comm.size();
    const int blockSize = static_cast<int>(Matrix::block_type::rows);
    const std::size_t totalBlkRows = matrix.N();
    const std::size_t totalBlkCols = matrix.M();
    std::size_t ownedBlkRows = interiorCellNum;

    // Collect global index information
    std::vector<int> ownedGlobals;
    std::vector<int> overlapGlobals;
    for (const auto& pair : comm->indexSet()) {
        if (pair.local().attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) {
            ownedGlobals.push_back(pair.global());
        } else {
            overlapGlobals.push_back(pair.global());
        }
    }
    if (ownedBlkRows == 0 && !ownedGlobals.empty()) {
        ownedBlkRows = ownedGlobals.size();
    }
    const std::size_t overlapBlkRows = totalBlkRows > ownedBlkRows ? (totalBlkRows - ownedBlkRows) : 0;

    std::sort(ownedGlobals.begin(), ownedGlobals.end());
    std::sort(overlapGlobals.begin(), overlapGlobals.end());

    // Map local overlap indices to owner neighbor ranks
    std::map<size_t, int> localToOwnerRank;
    for (auto it = comm->remoteIndices().begin(); it != comm->remoteIndices().end(); ++it) {
        int nbr = it->first;
        if (nbr == rank) continue;
        if (!it->second.first) continue;
        for (const auto& ri : *(it->second.first)) {
            if (ri.attribute() == Dune::OwnerOverlapCopyAttributeSet::owner) {
                localToOwnerRank[ri.localIndexPair().local().local()] = nbr;
            }
        }
    }

    std::map<int, std::size_t> nbr_bnd_nnz;
    std::map<int, std::size_t> nbr_overlap_cells;
    for (const auto& pair : localToOwnerRank) {
        nbr_overlap_cells[pair.second]++;
    }

    // Nonzero block analysis
    std::size_t int_nnz = 0;
    std::size_t bnd_nnz = 0;
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        auto r_idx = row.index();
        if (r_idx < ownedBlkRows) {
            for (auto col = row->begin(); col != row->end(); ++col) {
                if (col.index() < ownedBlkRows) {
                    ++int_nnz;
                } else {
                    ++bnd_nnz;
                    auto it_owner = localToOwnerRank.find(col.index());
                    if (it_owner != localToOwnerRank.end()) {
                        nbr_bnd_nnz[it_owner->second]++;
                    }
                }
            }
        }
    }

    // Neighbors
    std::vector<int> neighbors;
    for (int n : comm->remoteIndices().getNeighbours()) {
        if (n != rank) neighbors.push_back(n);
    }
    if (neighbors.empty()) {
        for (auto it = comm->remoteIndices().begin(); it != comm->remoteIndices().end(); ++it) {
            if (it->first != rank) {
                neighbors.push_back(it->first);
            }
        }
    }
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

    std::vector<std::string> nbrDetails;
    for (int n : neighbors) {
        std::size_t c_cells = nbr_overlap_cells[n];
        std::size_t c_nnz = nbr_bnd_nnz[n];
        nbrDetails.push_back(fmt::format("{} ({} ovlp blks, {} bnd nnz)", n, c_cells, c_nnz));
    }
    std::string nbrStr = "[" + fmt::format("{}", fmt::join(nbrDetails, ", ")) + "]";

    auto formatRanges = [](const std::vector<int>& ids) -> std::string {
        if (ids.empty()) return "none";
        std::vector<std::string> ranges;
        int start = ids[0];
        int prev = ids[0];
        for (size_t i = 1; i < ids.size(); ++i) {
            if (ids[i] == prev + 1) {
                prev = ids[i];
            } else {
                if (start == prev) ranges.push_back(std::to_string(start));
                else ranges.push_back(fmt::format("{}..{}", start, prev));
                start = ids[i];
                prev = ids[i];
            }
        }
        if (start == prev) ranges.push_back(std::to_string(start));
        else ranges.push_back(fmt::format("{}..{}", start, prev));

        std::string res;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (i > 0) res += ", ";
            res += ranges[i];
        }
        return res;
    };

    std::string ownedRangesStr = formatRanges(ownedGlobals);
    std::string overlapRangesStr = formatRanges(overlapGlobals);

    DeferredLogger local_logger;
    std::string msg = fmt::format(
        "Rank {:<2} | Local Block Rows: [0 .. {:>3}] owned ({:>3} blks), [{:>3} .. {:>3}] overlap ({:>3} blks) | Total: {:>3} blks\n"
        "        | Scalar Rows:     [0 .. {:>3}] owned ({:>3} eqs),  [{:>3} .. {:>3}] overlap ({:>3} eqs)  | Total: {:>3} eqs\n"
        "        | Global Cell IDs: [{}] owned, [{}] overlap\n"
        "        | Matrix NNZ:      {:>4} blks total ({:>4} int-to-int, {:>3} int-to-overlap boundary)\n"
        "        | Neighbors:       {}",
        rank,
        ownedBlkRows > 0 ? ownedBlkRows - 1 : 0, ownedBlkRows,
        ownedBlkRows, totalBlkRows > 0 ? totalBlkRows - 1 : 0, overlapBlkRows, totalBlkRows,
        ownedBlkRows > 0 ? ownedBlkRows * blockSize - 1 : 0, ownedBlkRows * blockSize,
        ownedBlkRows * blockSize, totalBlkRows > 0 ? totalBlkRows * blockSize - 1 : 0, overlapBlkRows * blockSize, totalBlkRows * blockSize,
        ownedRangesStr, overlapRangesStr,
        matrix.nonzeroes(), int_nnz, bnd_nnz,
        nbrStr
    );
    local_logger.info(msg);

    // Call collective reduction across ALL ranks
    const int totalOwnedBlks = basic_comm.sum(static_cast<int>(ownedBlkRows));
    auto global_logger = gatherDeferredLogger(local_logger, basic_comm);
    if (rank == 0) {
        OpmLog::info("\n================================================================================");
        OpmLog::info("                    MPI LINEAR SYSTEM MATRIX PARTITIONING");
        OpmLog::info("================================================================================");
        OpmLog::info(fmt::format("MPI Processes: {} | Block Size: {}x{} equations per cell", mpiSize, blockSize, blockSize));
        OpmLog::info("Matrix Storage: Distributed BCRSMatrix (Overlapping sub-matrices per MPI rank)");
        OpmLog::info("Ordering: OwnerCellsFirst = true (Owned equations come before overlap equations)");
        OpmLog::info("Linear Solver & Preconditioner: BiCGStab + ParallelOverlappingILU0");
        OpmLog::info("--------------------------------------------------------------------------------");
        global_logger.logMessages();
        OpmLog::info("--------------------------------------------------------------------------------");
        OpmLog::info(fmt::format("Total Active Reservoir Cells: {} blocks ({} scalar equations)", totalOwnedBlks, totalOwnedBlks * blockSize));
        OpmLog::info("Execution Mechanics:");
        OpmLog::info("  • ILU0 Preconditioner & Solves: Each rank ONLY factors & solves local owned rows [0 .. N_owned-1].");
        OpmLog::info("  • Overlap Rows [N_owned .. N_total-1]: Ghost rows synchronized across ranks via copyOwnerToAll.");
        OpmLog::info("================================================================================\n");
    }
}
#endif

template<class Matrix, class Vector, class Comm>
void FlexibleSolverInfo<Matrix,Vector,Comm>::create(const Matrix& matrix,
                                                    bool parallel,
                                                    const PropertyTree& prm,
                                                    std::size_t pressureIndex,
                                                    std::function<Vector()> weightsCalculator,
                                                    const bool forceSerial,
                                                    [[maybe_unused]] Comm* comm)
{
    // Write sizes of linear systems on all ranks to debug log.
    if (!forceSerial) {
#if HAVE_MPI
        auto basic_comm = comm->communicator();
#else
        auto basic_comm = Dune::Communication<Dune::No_Comm>{};
#endif // HAVE_MPI
        std::ostringstream os;
        os << "Linear system ";
        if (basic_comm.size() > 1) {
            os << fmt::format("on MPI rank: {} ", basic_comm.rank());
        }
        // The static_cast of Matrix::block_type::rows is needed for fmt version 10.
        // TODO: Check if the cast is still needed in future versions.
        os << fmt::format("blocksize: {} size: {:7d} block nonzeroes: {:9d}",
                          static_cast<int>(Matrix::block_type::rows), matrix.N(), matrix.nonzeroes());
        DeferredLogger local_logger;
        local_logger.debug(os.str());
        auto global_logger = gatherDeferredLogger(local_logger, basic_comm);
        if (basic_comm.rank() == 0) {
            global_logger.logMessages();
        }

#if HAVE_MPI
        if (parallel && comm) {
            logMatrixPartitioning(matrix, comm, this->interiorCellNum_);
        }
#endif
    }


    // Delete the operator and solver here to avoid
    // a large temporary memory peak from having two
    // operators and solvers at the same time.
    this->op_.reset();
    this->solver_.reset();

    if (parallel) {
#if HAVE_MPI
        if (!wellOperator_) {
            using ParOperatorType = Opm::GhostLastMatrixAdapter<Matrix, Vector, Vector, Comm>;
            auto pop = std::make_unique<ParOperatorType>(matrix, *comm);
            using FlexibleSolverType = Dune::FlexibleSolver<ParOperatorType>;
            auto sol = std::make_unique<FlexibleSolverType>(*pop, *comm, prm,
                                                            weightsCalculator,
                                                            pressureIndex);
            this->pre_ = &sol->preconditioner();
            this->op_ = std::move(pop);
            this->solver_ = std::move(sol);
        } else {
            using ParOperatorType = WellModelGhostLastMatrixAdapter<Matrix, Vector, Vector, true>;
            auto pop = std::make_unique<ParOperatorType>(matrix, *wellOperator_,
                                                         interiorCellNum_);
            using FlexibleSolverType = Dune::FlexibleSolver<ParOperatorType>;
            auto sol = std::make_unique<FlexibleSolverType>(*pop, *comm, prm,
                                                            weightsCalculator,
                                                            pressureIndex);
            this->pre_ = &sol->preconditioner();
            this->op_ = std::move(pop);
            this->solver_ = std::move(sol);
        }
#endif
    } else {
        if (!wellOperator_) {
            using SeqOperatorType = Dune::MatrixAdapter<Matrix, Vector, Vector>;
            auto sop = std::make_unique<SeqOperatorType>(matrix);
            using FlexibleSolverType = Dune::FlexibleSolver<SeqOperatorType>;
            auto sol = std::make_unique<FlexibleSolverType>(*sop, prm,
                                                            weightsCalculator,
                                                            pressureIndex);
            this->pre_ = &sol->preconditioner();
            this->op_ = std::move(sop);
            this->solver_ = std::move(sol);
        } else {
            using SeqOperatorType = WellModelMatrixAdapter<Matrix, Vector, Vector>;
            auto sop = std::make_unique<SeqOperatorType>(matrix, *wellOperator_);
            using FlexibleSolverType = Dune::FlexibleSolver<SeqOperatorType>;
            auto sol = std::make_unique<FlexibleSolverType>(*sop, prm,
                                                            weightsCalculator,
                                                            pressureIndex);
            this->pre_ = &sol->preconditioner();
            this->op_ = std::move(sop);
            this->solver_ = std::move(sol);
        }
    }
}

template<class Scalar, int Dim>
using BM = Dune::BCRSMatrix<MatrixBlock<Scalar,Dim,Dim>>;
template<class Scalar, int Dim>
using BV = Dune::BlockVector<Dune::FieldVector<Scalar,Dim>>;

#if HAVE_MPI
using CommunicationType = Dune::OwnerOverlapCopyCommunication<int,int>;
#else
using CommunicationType = Dune::Communication<int>;
#endif

#define INSTANTIATE_FLEX(T,Dim)                                                           \
    template void makeOverlapRowsInvalid<BM<T,Dim>>(BM<T,Dim>&, const std::vector<int>&); \
    template struct FlexibleSolverInfo<BM<T,Dim>,BV<T,Dim>,CommunicationType>;

#define INSTANTIATE_TYPE(T) \
    INSTANTIATE_FLEX(T,1)   \
    INSTANTIATE_FLEX(T,2)   \
    INSTANTIATE_FLEX(T,3)   \
    INSTANTIATE_FLEX(T,4)   \
    INSTANTIATE_FLEX(T,5)   \
    INSTANTIATE_FLEX(T,6)   \
    INSTANTIATE_FLEX(T,7)

INSTANTIATE_TYPE(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_TYPE(float)
#endif

}
}
