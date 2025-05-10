SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
BUILD_PATH="${SOURCE_DIR}/build"

${BUILD_PATH}/test-basic