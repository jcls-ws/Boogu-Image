#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export device="cuda:0"

# Tuned for a small-VRAM GPU (8 GB): without offload the pipeline pages weights into
# host RAM and the process gets OOM-killed. `sequential` keeps VRAM under 2 GB.
# Override per run, e.g. `HEIGHT=1024 WIDTH=1024 OFFLOAD=none bash test_turbo_fp8.sh`.
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
OFFLOAD="${OFFLOAD:-sequential}"

case "$OFFLOAD" in
    sequential) offload_args=(--enable_sequential_cpu_offload_flag True) ;;
    model)      offload_args=(--enable_model_cpu_offload_flag True) ;;
    group)      offload_args=(--enable_group_offload_flag True) ;;
    none)       offload_args=() ;;
    *)
        echo "Invalid OFFLOAD=$OFFLOAD (use sequential|model|group|none)" >&2
        exit 1
        ;;
esac

mkdir -p outputs/test_turbo_fp8_offload/

python inference_turbo.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Turbo-fp8" \
    --use_fp8_weights True \
    "${offload_args[@]}" \
    --instruction "一幅国风琉金风格的山水画作，展现了桂林山水在金光普照下的壮丽景象。远山层叠，江水如镜，山峰边缘勾勒着发光的金色线条。画面采用石青石绿岩彩与鎏金质感相结合，局部有厚涂油画笔触，空中飘浮着金色粒子，营造出梦幻朦胧而又磅礴大气的意境。" \
    --height "$HEIGHT" --width "$WIDTH" \
    --output_image_path "outputs/test_turbo_fp8_offload/out_1.png" \
    --device "$device"
