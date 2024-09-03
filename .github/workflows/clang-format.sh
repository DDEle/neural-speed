source .github/workflows/change_color.sh

pip install clang-format==12.*
LOG_DIR="logs"
LOG_PATH=${LOG_DIR}/clang-format.log
mkdir -p ${LOG_DIR}

python tools/clang-format.py

echo "run git diff"
git diff 2>&1 | tee -a ${LOG_PATH}

if [[ ! -f ${LOG_PATH} ]] || [[ $(grep -c "diff" ${LOG_PATH}) != 0 ]]; then
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, check passed!" && $LIGHT_PURPLE && echo "You can click on the artifact button to see the log details." && $RESET
exit 0
