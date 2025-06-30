#!/bin/bash

input_dir="$1"
results_dir_name="$2"
ckpt_path="$3"
video_dir="${input_dir}/Videos"
audio_dir="${input_dir}/Audios"
results_dir="${input_dir}/${results_dir_name}"

mkdir -p "$results_dir"

# Check if video directory exists and contains MP4 files
if [ ! -d "$video_dir" ]; then
    echo "Error: Video directory '${video_dir}' does not exist!"
    exit 1
fi

if [ ! "$(ls -A "${video_dir}"/*.mp4 2>/dev/null)" ]; then
    echo "Error: No MP4 files found in '${video_dir}'"
    exit 1
fi

for video_file in "${video_dir}"/*.mp4; do
    base_name=$(basename "$video_file" .mp4)
    # audio_file="${audio_dir}/${base_name}.wav"
    audio_file="assets/demo1_audio.wav"
    video_out_file="${results_dir}/${base_name}.mp4"
    
    if [[ -f "$video_out_file" ]]; then
        echo "Result for ${base_name} already exists, skipping..."
        continue
    fi

    if [[ -f "$audio_file" ]]; then
        uv run -m scripts.inference \
            --unet_config_path "configs/unet/stage2.yaml" \
            --inference_ckpt_path "${ckpt_path}" \
            --inference_steps 20 \
            --guidance_scale 1 \
            --video_path "$video_file" \
            --audio_path "$audio_file" \
            --video_out_path "$video_out_file"
    else
        echo "Audio file for ${base_name} not found, skipping..."
    fi
done