CONFIG_DIR="configs/hparam-search"

for config_file in "$CONFIG_DIR"/*.yaml; do
    model_version=$(basename "$config_file" .yaml)
    echo "Training model version: $model_version"
    uv run train --config "$CONFIG_DIR/$model_version.yaml"

    echo "Evaluating model version: $model_version"
    uv run eval --ckpt "chkp/$model_version/best_loss.ckpt"
done
