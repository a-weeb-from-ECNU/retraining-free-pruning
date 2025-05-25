#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

DIR="$1"
DIR_NAME=$(basename "$DIR")

# ==== 指定要跳过的目录名（basename，不含路径） ====
EXCLUDE_DIRS=("mnli" "squad" "squad_v2" "stsb")  # 举例：跳过 squad 和 stsb

# ==== 判断某个值是否在数组中 ====
function is_excluded() {
  local value="$1"
  for exclude in "${EXCLUDE_DIRS[@]}"; do
    if [[ "$value" == "$exclude" ]]; then
      return 0  # 在排除列表中
    fi
  done
  return 1  # 不在排除列表中
}

# ==== 遍历每个子目录 ====
for folder in "$DIR"/*; do
  if [ -d "$folder" ]; then
    task_name=$(basename "$folder")

    if is_excluded "$task_name"; then
      echo "Skipping excluded task: $task_name"
      continue
    fi

    echo -e "\n\n\033[1;34mRunning task: $task_name\033[0m"
    python3 test.py --model_name "$DIR_NAME" \
                    --task_name "$task_name" \
                    --ckpt_dir "$folder" \
                    --constraint 0.9 \
                    --seed 0
  else
    echo "Skipping non-directory: $folder"
  fi
done
