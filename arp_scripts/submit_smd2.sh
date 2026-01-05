#!/bin/bash

usage() {
cat << EOF
$(basename "$0"):
    Primary script to launch a smalldata_tools run analysis for LCLS-II.

    OPTIONS:
        -h|--help
            Definition of options
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            usage
            exit 0
            ;;
        -e|--experiment)
            EXP="$2"
            POSITIONAL+=("--experiment" "$2")
            shift 2
            ;;
        -r|--run)
            RUN="$2"
            POSITIONAL+=("--run" "$2")
            shift 2
            ;;
        -d|--directory)
            POSITIONAL+=("--directory" "$2")
            shift 2
            ;;
        -n|--nevents)
            NEVENTS="$2"
            POSITIONAL+=("--nevents" "$2")
            shift 2
            ;;
        --interactive)
            INTERACTIVE=1
            shift
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --eb_cores)
            EB_CORES="$2"
            shift 2
            ;;
        --srv_cores)
            SRV_CORES="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --reservation)
            RESERVATION="$2"
            shift 2
            ;;
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        --s3df)
            FORCE_S3DF=1
            shift
            ;;
        --mpi_optim)
            MPI_OPTIM=1
            POSITIONAL+=("--mpi_optim")
            shift
            ;;
        --psplot_live)
            PSPLOT_LIVE=1
            POSITIONAL+=("--psplot_live_mode")
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Replace argv with the properly-split positional args
set -- "${POSITIONAL[@]}"

umask 002  # new files/dirs permission mask

# Defaults from environment if submitted from elog
EXP="${EXPERIMENT:=$EXP}"
RUN="${RUN_NUM:=$RUN}"
HUTCH=${EXP:0:3}
ARP_LOCATION="${ARP_LOCATION:=LOCAL}"

# Export EXP and RUN for downstream scripts
export EXPERIMENT="$EXP"
export RUN_NUM="$RUN"

# Export useful path
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export SMD_ROOT
SMD_ROOT="$(echo "$MYDIR" | sed "s|/arp_scripts||g")"

export SIT_ENV_DIR="/sdf/group/lcls/ds/ana"

# Source env. Needed to get python
echo "Sourcing LCLS-II environment"
source "$SIT_ENV_DIR/sw/conda2/manage/bin/psconda.sh"

# Export xtcpp paths
XTCPP_DIR="$SMD_ROOT/../xtcpp"
if [ -d "$XTCPP_DIR/install" ]; then
    PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    XTCPP_PYTHON_PATH="$XTCPP_DIR/install/lib/python${PYTHON_VER}/site-packages"
    XTCPP_BIN_PATH="$XTCPP_DIR/install/bin"

    if [ -d "$XTCPP_PYTHON_PATH" ]; then
        export PYTHONPATH="$XTCPP_PYTHON_PATH:${PYTHONPATH}"
        echo "Added xtcpp Python bindings to PYTHONPATH: $XTCPP_PYTHON_PATH"
    fi

    if [ -d "$XTCPP_BIN_PATH" ]; then
        export PATH="$XTCPP_BIN_PATH:${PATH}"
        echo "Added xtcpp binaries to PATH: $XTCPP_BIN_PATH"
    fi

    export XTCPP_MPIDS_IDXMODE="${XTCPP_MPIDS_IDXMODE:-FAST}"
else
    echo "Warning: xtcpp install directory not found at $XTCPP_DIR/install"
    echo "         xtcpp may not be built or installed correctly"
fi

# Figure out the right base path for the data (or use S3DF in force case)
if [ -v FORCE_S3DF ]; then
    DATAPATH="/sdf/data/lcls/ds"
else
    DATAPATH=$(python "$SMD_ROOT/arp_scripts/file_location.py" -e "$EXP" -r "$RUN")
fi
export SIT_PSDM_DATA="$DATAPATH"

# Interactive mode
if [ -v INTERACTIVE ]; then
    export PS_SRV_NODES=1
    export PS_EB_NODES=1
    echo "ARGS being passed to run_smd2.sh:"
    printf '  [%q]\n' "$@"
    "$SMD_ROOT/arp_scripts/run_smd2.sh" "$@"
    exit 0
fi

# SLURM / MPI parameters
DEFPARTITION='milano'
PARTITION="${PARTITION:=$DEFPARTITION}"
ACCOUNT="${ACCOUNT:=lcls:$EXP}"

# EB, BD and SRV core allocation
if [[ "$PARTITION" == "milano" ]]; then
    CORES_PER_NODE=120
    DEFAULT_NODES=2
    NODES="${NODES:=$DEFAULT_NODES}"

    if [ -v MPI_OPTIM ]; then
        MPI_SLOTS=$((CORES_PER_NODE*(NODES-1)))      # all nodes but the one for SMD0
        DEFAULT_SRV_CORES=$((16*(NODES-1)))          # 16 writers per node minus the SMD0 node
    else
        MPI_SLOTS=$((CORES_PER_NODE*NODES-1))        # total minus SMD0
        DEFAULT_SRV_CORES=$((16*NODES))              # 16 writers per node
    fi

    if [ -v PSPLOT_LIVE ]; then
        DEFAULT_SRV_CORES=1
    fi

    SRV_CORES="${SRV_CORES:=$DEFAULT_SRV_CORES}"

    DEFAULT_EB_CORES=$(((MPI_SLOTS-SRV_CORES)/16))   # recommended ratio
    # BUGFIX: was assigning EB_CORES from EB_NODES; should use EB_CORES
    EB_CORES="${EB_CORES:=$DEFAULT_EB_CORES}"
fi

export PS_SRV_NODES="$SRV_CORES"
export PS_EB_NODES="$EB_CORES"

# If we run into the chunk size error:
# export PS_SMD_CHUNKSIZE=1073741824

SBATCH_ARGS=(--nodes "$NODES" --account "$ACCOUNT" -p "$PARTITION")
if [ -n "$RESERVATION" ]; then
    SBATCH_ARGS+=(--reservation "$RESERVATION")
fi

echo "sbatch arguments:"
printf '  %q' sbatch "${SBATCH_ARGS[@]}" "$SMD_ROOT/arp_scripts/run_smd2.sh" "$@"
echo
echo "ARGS being passed to run_smd2.sh:"
printf '  [%q]\n' "$@"

sbatch "${SBATCH_ARGS[@]}" "$SMD_ROOT/arp_scripts/run_smd2.sh" "$@"
