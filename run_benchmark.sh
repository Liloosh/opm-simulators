#!/usr/bin/env bash
# ==============================================================================
# SPE11C Simulation & HPC Solver Benchmark Runner
#
# This standalone script runs one or multiple SPE11C benchmark cases using OPM Flow
# with the BiCGSTAB linear solver and our integrated Dune-ISTL HPC solver profiler.
#
# Parameters:
#   -p, -np, --procs, --processes <N>
#       Number of MPI processes (default: 4)
#   -t, --threads, --threads-per-process <N>
#       Number of OpenMP threads per process (default: 2)
#   -s, --precond, --preconditioner <TYPE>
#       Preconditioner type (default: ilu0).
#       Supported: ilu0, cpr, dilu, amg, cprw, cpr_trueimpes, cpr_quasiimpes
#   -c, --case, --cases, --grid <CASES>
#       SPE11C case resolution(s) or .DATA path(s).
#       Accepts:
#         - Single case: -c RSC_16
#         - Comma-separated: -c RSC_16,RSC_32,RSC_64
#         - Space-separated: -c "16 32 64"
#         - Multiple flags: -c RSC_16 -c RSC_32
#         - All cases: -c all or --all-cases
#       Default: RSC_16
#   --all-cases, --all
#       Run all 20 SPE11C resolution cases (RSC_16 to RSC_320)
#   -o, --output-dir <DIR>
#       Simulation output directory base (default: ./simulation_output)
#   --profile-dir <DIR>
#       Base directory for solver profiler CSV and report files (default: ./benchmark_runs)
#   --dry-run
#       Perform a dry-run check (deck reading, partitioning) without time-stepping
#   --stop-on-error
#       Stop the batch immediately if any dataset simulation fails
#   -h, --help
#       Display this help message and exit
#
# Any additional options (e.g. --linear-solver-max-iter=50) will be passed directly to flow.
# ==============================================================================

set -uo pipefail

# Avoid hwloc OpenGL / OpenCL scan hangs on systems without active display/GPU context
export HWLOC_COMPONENTS="-gl,-opencl"

# ----------------- Default Parameters -----------------
PROCS=4
THREADS=2
PRECOND="ilu0"
RAW_CASES=()
ALL_CASES=false
CUSTOM_OUTPUT_DIR=""
PROFILE_BASE_DIR="benchmark_runs"
DRY_RUN=false
STOP_ON_ERROR=false
EXTRA_ARGS=()

# ----------------- Helper Functions -----------------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [-- ADDITIONAL_FLOW_OPTIONS]

Options:
  -p, -np, --procs <N>          Number of MPI processes [default: $PROCS]
  -t, --threads <N>            Number of OpenMP threads per process [default: $THREADS]
  -s, --precond <TYPE>         Preconditioner type [default: $PRECOND]
                               (ilu0, cpr, dilu, amg, cprw, cpr_trueimpes, cpr_quasiimpes)
  -c, --case, --cases <CASES>  One or multiple SPE11C cases or .DATA paths [default: RSC_16]
                               Supports:
                                 - Single: -c RSC_16
                                 - Comma-separated: -c RSC_16,RSC_32,RSC_48
                                 - Space-separated: -c "16 32 64"
                                 - Repeat flag: -c RSC_16 -c RSC_32
                                 - All available cases: -c all
  --all-cases, --all           Run all 20 SPE11C cases (RSC_16 to RSC_320)
  -o, --output-dir <DIR>       Directory for simulation output files
                               [default: ./simulation_output/<case>_<precond>_np<P>_t<T>]
  --profile-dir <DIR>          Base directory for solver profiler metrics
                               [default: $PROFILE_BASE_DIR]
  --dry-run                    Verify deck parsing and grid setup without time integration
  --stop-on-error              Abort batch execution if any case fails
  -h, --help                   Display this help message and exit

Examples:
  # Single case run
  $0 -p 4 -t 2 -s ilu0 -c RSC_16

  # Multiple cases in sequence
  $0 -p 4 -t 2 -s ilu0 -c RSC_16,RSC_32,RSC_48

  # Compare CPR on multiple cases
  $0 -p 8 -t 1 -s cpr -c "RSC_16 RSC_32"

  # Fast dry-run check across multiple datasets
  $0 -p 2 -t 2 -s ilu0 -c 16,32,64 --dry-run

  # Run all 20 resolutions
  $0 -p 4 -t 2 -s ilu0 --all-cases
EOF
    exit 0
}

# ----------------- Parse Command Line -----------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|-np|--procs|--processes)
            PROCS="$2"
            shift 2
            ;;
        --procs=*|--processes=*|-p=*|-np=*)
            PROCS="${1#*=}"
            shift
            ;;
        -t|--threads|--threads-per-process)
            THREADS="$2"
            shift 2
            ;;
        --threads=*|--threads-per-process=*|-t=*)
            THREADS="${1#*=}"
            shift
            ;;
        -s|--precond|--preconditioner)
            PRECOND="$2"
            shift 2
            ;;
        --precond=*|--preconditioner=*|-s=*)
            PRECOND="${1#*=}"
            shift
            ;;
        -c|--case|--cases|--grid)
            RAW_CASES+=("$2")
            shift 2
            ;;
        --case=*|--cases=*|--grid=*|-c=*)
            RAW_CASES+=("${1#*=}")
            shift
            ;;
        --all-cases|--all)
            ALL_CASES=true
            shift
            ;;
        -o|--output-dir)
            CUSTOM_OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*|-o=*)
            CUSTOM_OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --profile-dir)
            PROFILE_BASE_DIR="$2"
            shift 2
            ;;
        --profile-dir=*)
            PROFILE_BASE_DIR="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stop-on-error)
            STOP_ON_ERROR=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ----------------- Find Flow Executable -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLOW_BIN=""
CANDIDATE_FLOW_BINS=(
    "${OPM_FLOW_BIN:-}"
    "${SCRIPT_DIR}/build-cmake/bin/flow"
    "${SCRIPT_DIR}/build/bin/flow"
    "${HOME}/opm-simulators/build-cmake/bin/flow"
    "${HOME}/opm-simulators/build/bin/flow"
    "$(which flow 2>/dev/null || true)"
)

for candidate in "${CANDIDATE_FLOW_BINS[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        FLOW_BIN="$candidate"
        break
    fi
done

if [[ -z "$FLOW_BIN" ]]; then
    echo "ERROR: Could not find 'flow' executable." >&2
    echo "Please build flow or set OPM_FLOW_BIN environment variable." >&2
    exit 1
fi

# ----------------- Find SPE11C Dataset Root -----------------
DATA_ROOT=""
CANDIDATE_DATA_DIRS=(
    "${SPE11C_DATA_DIR:-}"
    "${HOME}/spe11c_docker_and_slurm"
    "${SCRIPT_DIR}/spe11c_docker_and_slurm"
    "${SCRIPT_DIR}/../spe11c_docker_and_slurm"
    "./spe11c_docker_and_slurm"
)

for candidate in "${CANDIDATE_DATA_DIRS[@]}"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
        DATA_ROOT="$(realpath "$candidate")"
        break
    fi
done

# Standard known resolutions in SPE11C
ALL_AVAILABLE_CASES=(
    "RSC_16"  "RSC_32"  "RSC_48"  "RSC_64"  "RSC_80"
    "RSC_96"  "RSC_112" "RSC_128" "RSC_144" "RSC_160"
    "RSC_176" "RSC_192" "RSC_208" "RSC_224" "RSC_240"
    "RSC_256" "RSC_272" "RSC_288" "RSC_304" "RSC_320"
)

# ----------------- Resolve Case List -----------------
TARGET_CASES=()

if [[ "$ALL_CASES" == true ]]; then
    TARGET_CASES=("${ALL_AVAILABLE_CASES[@]}")
elif [[ ${#RAW_CASES[@]} -eq 0 ]]; then
    TARGET_CASES=("RSC_16")
else
    for item in "${RAW_CASES[@]}"; do
        # Replace commas with spaces and iterate
        cleaned="${item//,/ }"
        for token in $cleaned; do
            token_upper="$(echo "$token" | tr '[:lower:]' '[:upper:]')"
            if [[ "$token_upper" == "ALL" ]]; then
                TARGET_CASES=("${ALL_AVAILABLE_CASES[@]}")
                break 2
            fi
            TARGET_CASES+=("$token")
        done
    done
fi

# Remove duplicates while preserving order
UNIQUE_CASES=()
for c in "${TARGET_CASES[@]}"; do
    found=false
    for u in "${UNIQUE_CASES[@]}"; do
        if [[ "$u" == "$c" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        UNIQUE_CASES+=("$c")
    fi
done
TARGET_CASES=("${UNIQUE_CASES[@]}")

TOTAL_CASES=${#TARGET_CASES[@]}
if [[ $TOTAL_CASES -eq 0 ]]; then
    echo "ERROR: No target cases specified." >&2
    exit 1
fi

# ----------------- Resolve .DATA Deck Paths -----------------
RESOLVED_CASE_NAMES=()
RESOLVED_DECK_PATHS=()

for case_arg in "${TARGET_CASES[@]}"; do
    data_file=""
    case_name=""

    if [[ -f "$case_arg" ]]; then
        data_file="$(realpath "$case_arg")"
        case_name="$(basename "$case_arg" .DATA)"
    else
        # Normalize string: "16" -> "RSC_16", "rsc_16" -> "RSC_16"
        c_clean="$(echo "$case_arg" | tr '[:lower:]' '[:upper:]')"
        if [[ ! "$c_clean" =~ ^RSC_ ]]; then
            c_clean="RSC_${c_clean}"
        fi
        case_name="$c_clean"

        if [[ -n "$DATA_ROOT" && -f "${DATA_ROOT}/${case_name}/deck/${case_name}.DATA" ]]; then
            data_file="${DATA_ROOT}/${case_name}/deck/${case_name}.DATA"
        elif [[ -f "${case_name}/deck/${case_name}.DATA" ]]; then
            data_file="$(realpath "${case_name}/deck/${case_name}.DATA")"
        fi
    fi

    if [[ -z "$data_file" || ! -f "$data_file" ]]; then
        echo "ERROR: Could not locate DATA deck for case '$case_arg'." >&2
        if [[ -n "$DATA_ROOT" ]]; then
            echo "Looked in: ${DATA_ROOT}/${case_name}/deck/${case_name}.DATA" >&2
        fi
        exit 1
    fi

    RESOLVED_CASE_NAMES+=("$case_name")
    RESOLVED_DECK_PATHS+=("$data_file")
done

# ----------------- OpenMP & MPI Affinity Setup -----------------
export OMP_NUM_THREADS="$THREADS"
mkdir -p "$PROFILE_BASE_DIR"

# Detect hardware CPU topology
PHYS_CORES=$(grep -m1 "cpu cores" /proc/cpuinfo 2>/dev/null | awk '{print $4}')
TOTAL_VCPUS=$(nproc --all 2>/dev/null || grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo 1)
if [[ -z "$PHYS_CORES" || "$PHYS_CORES" -le 0 ]]; then
    PHYS_CORES=$TOTAL_VCPUS
fi

TOTAL_EXEC_THREADS=$((PROCS * THREADS))
MPI_LAUNCH_ARGS=()
AFFINITY_NOTE=""

if [[ $PROCS -le $PHYS_CORES && $THREADS -eq 1 ]]; then
    MPI_LAUNCH_ARGS=(--bind-to core -np "$PROCS")
    AFFINITY_NOTE="Clean physical core allocation (--bind-to core; zero oversubscription)"
elif [[ $TOTAL_EXEC_THREADS -le $TOTAL_VCPUS ]]; then
    if [[ $TOTAL_EXEC_THREADS -le $PHYS_CORES ]]; then
        MPI_LAUNCH_ARGS=(--map-by ppr:${PROCS}:node:PE=${THREADS} --bind-to core -np "$PROCS")
        AFFINITY_NOTE="Hybrid MPI+OpenMP physical core allocation (PE=${THREADS}; non-overlapping cores)"
    else
        MPI_LAUNCH_ARGS=(--bind-to none -np "$PROCS")
        AFFINITY_NOTE="Hybrid MPI+OpenMP allocation (--bind-to none; ${TOTAL_EXEC_THREADS} threads on ${TOTAL_VCPUS} vCPUs)"
    fi
else
    MPI_LAUNCH_ARGS=(--oversubscribe --bind-to none -np "$PROCS")
    AFFINITY_NOTE="Oversubscribed (${TOTAL_EXEC_THREADS} threads > ${TOTAL_VCPUS} vCPUs; high profiling noise)"
fi

# ----------------- Batch Summary Header -----------------
BATCH_START_TIME=$(date +%s)
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "                   SPE11C MULTI-DATASET BENCHMARK EXECUTION"
echo "================================================================================"
echo " Total Datasets:        ${TOTAL_CASES}"
echo " Case Resolutions:      ${RESOLVED_CASE_NAMES[*]}"
echo " Flow Binary:           ${FLOW_BIN}"
if [[ -n "$DATA_ROOT" ]]; then
echo " SPE11C Data Root:      ${DATA_ROOT}"
fi
echo " CPU Topology:          ${PHYS_CORES} Physical Cores / ${TOTAL_VCPUS} Logical vCPUs"
echo " MPI Processes (-np):   ${PROCS}"
echo " OpenMP Threads (-t):   ${THREADS} (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
echo " Core Allocation:       ${AFFINITY_NOTE}"
echo " Linear Solver:         BiCGSTAB"
echo " Preconditioner (-s):   ${PRECOND}"
echo " Profiler Output Dir:   ${PROFILE_BASE_DIR}"
if [[ "$DRY_RUN" == true ]]; then
echo " Execution Mode:        Dry-run (validating grids & Zoltan partitions)"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
echo " Additional Arguments:  ${EXTRA_ARGS[*]}"
fi
echo "================================================================================"
echo ""

# Tracking variables for batch summary
BATCH_RESULTS_CASE=()
BATCH_RESULTS_STATUS=()
BATCH_RESULTS_ELAPSED=()
BATCH_RESULTS_DIR=()

OVERALL_SUCCESS=0
OVERALL_FAILED=0

# ----------------- Main Execution Loop -----------------
for ((i=0; i<TOTAL_CASES; i++)); do
    CURRENT_CASE="${RESOLVED_CASE_NAMES[$i]}"
    CURRENT_DECK="${RESOLVED_DECK_PATHS[$i]}"
    STEP_NUM=$((i + 1))

    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo " [${STEP_NUM}/${TOTAL_CASES}] Running Dataset: ${CURRENT_CASE}"
    echo " Deck: ${CURRENT_DECK}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # Determine simulation output directory
    if [[ -n "$CUSTOM_OUTPUT_DIR" ]]; then
        if [[ $TOTAL_CASES -gt 1 ]]; then
            SIM_DIR="${CUSTOM_OUTPUT_DIR}/${CURRENT_CASE}_${PRECOND}_np${PROCS}_t${THREADS}"
        else
            SIM_DIR="${CUSTOM_OUTPUT_DIR}"
        fi
    else
        SIM_DIR="simulation_output/${CURRENT_CASE}_${PRECOND}_np${PROCS}_t${THREADS}"
    fi
    mkdir -p "$SIM_DIR"

    CMD=(
        mpirun "${MPI_LAUNCH_ARGS[@]}"
        "$FLOW_BIN"
        "$CURRENT_DECK"
        "--output-dir=${SIM_DIR}"
        "--threads-per-process=${THREADS}"
        "--linear-solver=${PRECOND}"
        "--enable-solver-profiling=true"
        "--solver-profile-print-summary=true"
        "--solver-profile-dir=${PROFILE_BASE_DIR}"
    )

    if [[ "$DRY_RUN" == true ]]; then
        CMD+=("--enable-dry-run=true")
    fi

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        CMD+=("${EXTRA_ARGS[@]}")
    fi

    RUN_START=$(date +%s)
    "${CMD[@]}"
    RUN_STATUS=$?
    RUN_END=$(date +%s)
    RUN_ELAPSED=$((RUN_END - RUN_START))

    # Detect session directory created during this specific run
    LATEST_SESSION="N/A"
    for sdir in $(ls -dt "${PROFILE_BASE_DIR}"/run_* 2>/dev/null); do
        if [[ -d "$sdir" ]]; then
            DIR_MTIME=$(stat -c %Y "$sdir" 2>/dev/null || stat -f %m "$sdir" 2>/dev/null || echo 0)
            if [[ $DIR_MTIME -ge $RUN_START ]]; then
                LATEST_SESSION="$sdir"
                break
            fi
        fi
    done

    if [[ $RUN_STATUS -eq 0 ]]; then
        STATUS_STR="SUCCESS"
        OVERALL_SUCCESS=$((OVERALL_SUCCESS + 1))
        echo "--> [${CURRENT_CASE}] Completed successfully in ${RUN_ELAPSED}s."
    else
        STATUS_STR="FAILED(exit:${RUN_STATUS})"
        OVERALL_FAILED=$((OVERALL_FAILED + 1))
        echo "--> [${CURRENT_CASE}] FAILED with exit code ${RUN_STATUS} after ${RUN_ELAPSED}s."
    fi

    BATCH_RESULTS_CASE+=("$CURRENT_CASE")
    BATCH_RESULTS_STATUS+=("$STATUS_STR")
    BATCH_RESULTS_ELAPSED+=("${RUN_ELAPSED}s")
    BATCH_RESULTS_DIR+=("${LATEST_SESSION:-N/A}")

    echo ""

    if [[ $RUN_STATUS -ne 0 && "$STOP_ON_ERROR" == true ]]; then
        echo "ERROR: Aborting remaining datasets due to --stop-on-error." >&2
        break
    fi
done

BATCH_END_TIME=$(date +%s)
TOTAL_BATCH_ELAPSED=$((BATCH_END_TIME - BATCH_START_TIME))

# ----------------- Print Batch Summary Table -----------------
echo "================================================================================"
echo "                   BATCH BENCHMARK SUMMARY REPORT"
echo "================================================================================"
printf "%-4s  %-12s  %-14s  %-12s  %-s\n" "#" "Dataset" "Status" "Time" "Profiler Run Directory"
echo "--------------------------------------------------------------------------------"
for ((i=0; i<${#BATCH_RESULTS_CASE[@]}; i++)); do
    printf "%-4s  %-12s  %-14s  %-12s  %-s\n" \
        "$((i+1))" \
        "${BATCH_RESULTS_CASE[$i]}" \
        "${BATCH_RESULTS_STATUS[$i]}" \
        "${BATCH_RESULTS_ELAPSED[$i]}" \
        "${BATCH_RESULTS_DIR[$i]}"
done
echo "================================================================================"
echo " Summary: Total: ${TOTAL_CASES} | Succeeded: ${OVERALL_SUCCESS} | Failed: ${OVERALL_FAILED} | Total Time: ${TOTAL_BATCH_ELAPSED}s"
echo "================================================================================"

# ----------------- Save Batch Summary to File -----------------
BATCH_REPORT_TXT="${PROFILE_BASE_DIR}/batch_summary_${BATCH_TIMESTAMP}.txt"
BATCH_REPORT_CSV="${PROFILE_BASE_DIR}/batch_summary_${BATCH_TIMESTAMP}.csv"

{
    echo "SPE11C BATCH BENCHMARK REPORT"
    echo "Date: $(date)"
    echo "Preconditioner: ${PRECOND}"
    echo "MPI Processes: ${PROCS}"
    echo "OpenMP Threads: ${THREADS}"
    echo "Total Time: ${TOTAL_BATCH_ELAPSED}s"
    echo ""
    printf "%-4s  %-12s  %-14s  %-12s  %-s\n" "#" "Dataset" "Status" "Time" "Profiler Run Directory"
    echo "--------------------------------------------------------------------------------"
    for ((i=0; i<${#BATCH_RESULTS_CASE[@]}; i++)); do
        printf "%-4s  %-12s  %-14s  %-12s  %-s\n" \
            "$((i+1))" \
            "${BATCH_RESULTS_CASE[$i]}" \
            "${BATCH_RESULTS_STATUS[$i]}" \
            "${BATCH_RESULTS_ELAPSED[$i]}" \
            "${BATCH_RESULTS_DIR[$i]}"
    done
} > "$BATCH_REPORT_TXT"

{
    echo "index,dataset,status,time_seconds,profiler_dir"
    for ((i=0; i<${#BATCH_RESULTS_CASE[@]}; i++)); do
        echo "$((i+1)),${BATCH_RESULTS_CASE[$i]},${BATCH_RESULTS_STATUS[$i]},${BATCH_RESULTS_ELAPSED[$i]%s},${BATCH_RESULTS_DIR[$i]}"
    done
} > "$BATCH_REPORT_CSV"

echo "Batch report files saved:"
echo "  • Text: ${BATCH_REPORT_TXT}"
echo "  • CSV:  ${BATCH_REPORT_CSV}"
echo ""

if [[ $OVERALL_FAILED -gt 0 ]]; then
    exit 1
fi
exit 0
