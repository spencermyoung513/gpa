"""A script to evaluate a baseline heuristic-based link predictor."""
import argparse
from pathlib import Path
from pprint import pprint
from typing import Literal

import torch
import yaml
from gpa.common.helpers import connect_products_with_nearest_price_tag
from gpa.common.helpers import connect_products_with_nearest_price_tag_below
from gpa.common.helpers import connect_products_with_nearest_price_tag_per_group
from gpa.common.helpers import get_candidate_edges
from gpa.common.helpers import parse_into_subgraphs
from gpa.datamodules import PriceAttributionDataModule
from gpa.datasets.attribution import DetectionGraph
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.wrappers import BootStrapper
from tqdm import tqdm


def evaluate(
    dataset_dir: Path,
    results_dir: Path,
    method: Literal["nearest", "nearest_below", "nearest_per_group"],
):
    if method == "nearest":
        connector = connect_products_with_nearest_price_tag
    elif method == "nearest_below":
        connector = connect_products_with_nearest_price_tag_below
    elif method == "nearest_per_group":
        connector = connect_products_with_nearest_price_tag_per_group
    else:
        raise ValueError(f"Unknown method: {method}")

    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    datamodule = PriceAttributionDataModule(
        data_dir=dataset_dir,
        num_workers=0,
    )
    datamodule.setup("")

    precision = BootStrapper(BinaryPrecision())
    recall = BootStrapper(BinaryRecall())
    f1 = BootStrapper(BinaryF1Score())

    dataloader = datamodule.test_dataloader()
    for batch in tqdm(dataloader, desc="Evaluating heuristic..."):
        real_edges, fake_edges = get_candidate_edges(batch, balanced=False)
        kwargs = {
            "centroids": batch.x[
                :, DetectionGraph.BBOX_START_IDX : DetectionGraph.BBOX_END_IDX + 3
            ],
            "product_indices": batch.product_indices,
            "price_indices": batch.price_indices,
        }
        if method == "nearest_per_group":
            kwargs["cluster_assignment"] = parse_into_subgraphs(
                batch.shared_upc_edge_index, num_nodes=len(batch.x)
            )
        pred_edge_index = connector(**kwargs)

        real_edges_set = set(map(tuple, real_edges.T.tolist()))
        fake_edges_set = set(map(tuple, fake_edges.T.tolist()))
        pred_edges_set = set(map(tuple, pred_edge_index.T.tolist()))

        real_edge_probs = torch.tensor(
            [float(edge in pred_edges_set) for edge in real_edges_set]
        )
        fake_edge_probs = torch.tensor(
            [float(edge in pred_edges_set) for edge in fake_edges_set]
        )

        for probs, targets in [
            (real_edge_probs, torch.ones_like(real_edge_probs)),
            (fake_edge_probs, torch.zeros_like(fake_edge_probs)),
        ]:
            precision.update(probs, targets)
            recall.update(probs, targets)
            f1.update(probs, targets)

    metrics = {
        "precision": {k: v.item() for k, v in precision.compute().items()},
        "recall": {k: v.item() for k, v in recall.compute().items()},
        "f1": {k: v.item() for k, v in f1.compute().items()},
    }
    results_path = results_dir / "eval_metrics.yaml"
    with open(results_path, "w") as f:
        yaml.dump(metrics, f)

    pprint(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--method", choices=["nearest", "nearest_below", "nearest_per_group"]
    )
    args = parser.parse_args()
    evaluate(args.dataset_dir, args.results_dir, args.method)


if __name__ == "__main__":
    main()
