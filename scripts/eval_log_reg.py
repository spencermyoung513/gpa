"""A script to evaluate a baseline logistic-regression-based link predictor."""
import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from gpa.datasets.attribution import DetectionGraph
from gpa.datasets.attribution import PriceAttributionDataset
from gpa.training.transforms import MakeBoundingBoxTranslationInvariant
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


def dataset_to_X_y(dataset: PriceAttributionDataset):
    X, y = [], []
    for graph in tqdm(dataset, desc="Building X, y"):
        graph: DetectionGraph
        gt_edges = set(map(tuple, graph.gt_prod_price_edge_index.T.tolist()))
        src, dst = torch.cartesian_prod(graph.product_indices, graph.price_indices).T
        for src_idx, dst_idx in zip(src, dst):
            bbox_indices = slice(graph.BBOX_START_IDX, graph.BBOX_END_IDX + 1)
            src_bbox = graph.x[src_idx, bbox_indices].numpy()
            dst_bbox = graph.x[dst_idx, bbox_indices].numpy()
            X.append(np.concatenate([src_bbox, dst_bbox]))
            edge_is_real = (src_idx.item(), dst_idx.item()) in gt_edges or (
                dst_idx.item(),
                src_idx.item(),
            ) in gt_edges
            y.append(int(edge_is_real))
    return np.vstack(X), np.array(y)


def main(
    dataset_dir: Path,
    results_dir: Path,
    invariant_centroids: bool = False,
):
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    transform = MakeBoundingBoxTranslationInvariant() if invariant_centroids else None
    train_dataset = PriceAttributionDataset(
        root=dataset_dir / "train", transform=transform
    )
    test_dataset = PriceAttributionDataset(
        root=dataset_dir / "test", transform=transform
    )
    X_train, y_train = dataset_to_X_y(train_dataset)
    X_test, y_test = dataset_to_X_y(test_dataset)

    model = LogisticRegression()
    print("Fitting model")
    model.fit(X_train, y_train)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=y_test,
        y_pred=model.predict(X_test),
        average="binary",
        pos_label=1,
    )
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    results_path = results_dir / "eval_metrics.yaml"
    with open(results_path, "w") as f:
        yaml.dump(metrics, f)

    pprint(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--invariant-centroids", action="store_true")
    args = parser.parse_args()
    main(args.dataset_dir, args.results_dir, args.invariant_centroids)
