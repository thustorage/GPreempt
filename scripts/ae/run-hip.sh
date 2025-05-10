SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
BUILD_PATH="${SOURCE_DIR}/build"
CONFIG_PATH="${SOURCE_DIR}/config"
RESULT_PATH="${1:-${SOURCE_DIR}/results}"

configs="A B C D E REAL Y Z"
clients="rtonlyclient seqclient baseclient blpclient gpreemptclient"

mkdir -p "${RESULT_PATH}/data"

for config in $configs; do
    for client in $clients; do
        echo "Running $client with config $config"
        ssh 10.0.2.190 "${BUILD_PATH}/${client} ${CONFIG_PATH}/short/${config}.json" > ${RESULT_PATH}/data/hip-${client}-${config}.json
        sleep 10
    done
done
