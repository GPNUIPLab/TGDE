#!/bin/bash
# 方案 A (C23_out128 联合融合) 一键运行脚本。
# 用法:
#   bash run_a.sh <BM_llm|BM_rake|IT_llm|IT_rake> [device]
# 例如:  bash run_a.sh BM_llm 0
#   BM = 生物医药专利 (原 Patent)   IT = 智能交通论文 (原 Rawdata)
#   llm / rake = 概念抽取方式（llm 为大语言模型抽取，rake 为关键词抽取）
# 输出: code/<BM|IT>/pipeline/output/experiments/fusion_concat/<EXP>_s<seed>/
#   (model.pt.best, metrics.json, predictions.pkl.gz, train.log, eval.log, config.yaml)
set -eu
TGDE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?用法: run_a.sh <BM_llm|BM_rake|IT_llm|IT_rake> [device]}"
DEV="${2:-0}"
SEED="${SEED:-42}"

PY=${PY:-python}
PYTHONPATH="$TGDE/code"
export PYTHONPATH

case "$NAME" in
  BM_llm|bm_llm)   PIPE="BM"; CFG="pipeline/experiments/fusion_c23_out128.yaml";      EXP="C23_out128_lr_2e-4" ;;
  BM_rake|bm_rake) PIPE="BM"; CFG="pipeline/experiments/fusion_c23_rake_out128.yaml"; EXP="C23_rake_out128_lr_2e-4" ;;
  IT_llm|it_llm)   PIPE="IT"; CFG="pipeline/experiments/fusion_c23_out128.yaml";      EXP="C23_out128_lr_2e-4" ;;
  IT_rake|it_rake) PIPE="IT"; CFG="pipeline/experiments/fusion_c23_rake_out128.yaml"; EXP="C23_rake_out128_lr_2e-4" ;;
  *) echo "未知数据集: $NAME (可选: BM_llm BM_rake IT_llm IT_rake)" >&2; exit 1 ;;
esac

cd "$TGDE/code"
bash "$TGDE/link_datasets.sh"
echo "== 运行方案 A: $NAME (device=$DEV, seed=$SEED)"
exec $PY "$PIPE/$CFG" --experiments="$EXP" --seed="$SEED" --device="$DEV"
