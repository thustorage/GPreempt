SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
BUILD_PATH="${SOURCE_DIR}/build"
CONFIG_PATH="${SOURCE_DIR}/config"
RESULT_PATH="${1:-${SOURCE_DIR}/results}"

mkdir -p "${RESULT_PATH}/data"

configs="A1 A2 A3 A4 A5 A6"
clients="gpreemptclient"
for config in $configs; do
    for client in $clients; do
        echo "Running $client with config $config"
        ${BUILD_PATH}/${client} ${CONFIG_PATH}/micro-5.3/${config}.json | grep -v '^Init' > ${RESULT_PATH}/data/cuda-${client}-${config}.json
    done
done

configs="B0 B1 B2 B3 B4"
clients="blpclient gpreemptclient"
for config in $configs; do
    for client in $clients; do
        echo "Running $client with config $config"
        ${BUILD_PATH}/${client} ${CONFIG_PATH}/micro-3.2.2/${config}.json | grep -v '^Init' > ${RESULT_PATH}/data/cuda-${client}-${config}.json
    done
done

for config in $configs; do
    echo "Running gpreemptclient without scheduled pre-preemption with config $config"
    client="gpreemptclient"
    ${BUILD_PATH}/${client} ${CONFIG_PATH}/micro-3.2.2/${config}.json false | grep -v '^Init' > ${RESULT_PATH}/data/cuda-gpreemptclient_wo_res-${config}.json
done
