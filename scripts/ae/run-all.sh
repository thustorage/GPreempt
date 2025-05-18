SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

RESULT_PATH="${1:-${SOURCE_DIR}/results}"

SCRIPT_PATH="${SOURCE_DIR}/scripts/ae"
CONFIG_PATH="${SOURCE_DIR}/config"

mkdir -p "${RESULT_PATH}/data"
mkdir -p "${RESULT_PATH}/figures"

"${SCRIPT_PATH}/run-cuda.sh" "${RESULT_PATH}"
"${SCRIPT_PATH}/run-hip.sh" "${RESULT_PATH}"
"${SCRIPT_PATH}/run-micro.sh" "${RESULT_PATH}"

# Copy the results of reef to the data directory
# Because reef cannot run on ubuntu 22.04
# If AE reviewers want to run reef client, 
# please contact us through hotcrp and make an appointment with us

cp ${SOURCE_DIR}/reef_results/* ${RESULT_PATH}/data