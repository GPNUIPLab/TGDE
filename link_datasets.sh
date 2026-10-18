#!/bin/bash
# 将 TGDE/dataset 下的 4 个数据集表征链接到两条管线的 output/<concept_source> 目录,
# 使 run_experiment.py 无需修改即可按原始路径加载数据。
# 用法: bash link_datasets.sh   (幂等, 重复执行先删除旧链接)
set -eu
ROOT="$(cd "$(dirname "$0")/code" && pwd)"
DATA="$(cd "$(dirname "$0")/dataset" && pwd)"

declare -A MAP=(
  ["$ROOT/BM/pipeline/output/llm"]="$DATA/BM_llm"
  ["$ROOT/BM/pipeline/output/rake"]="$DATA/BM_rake"
  ["$ROOT/IT/pipeline/output/llm"]="$DATA/IT_llm"
  ["$ROOT/IT/pipeline/output/rake"]="$DATA/IT_rake"
)

for link in "${!MAP[@]}"; do
  mkdir -p "$(dirname "$link")"
  [ -L "$link" ] && rm "$link"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "跳过 $link (已存在真实目录)" >&2; continue
  fi
  ln -s "${MAP[$link]}" "$link"
  echo "linked: $link -> ${MAP[$link]}"
done
echo "done."
