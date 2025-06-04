"""This script converts our raw pricing info tables (from the labeling tool) into usable components for a graph representation."""
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from gpa.common.objects import GraphComponents
from gpa.common.objects import ProductPriceGroup
from gpa.common.objects import UPCGroup
from open_clip import CLIP
from PIL import Image
from tqdm import tqdm


def csv_array_str_to_list(x: str) -> list[str]:
    """Convert a string representation of an array in a CSV file to an actual list."""
    # Handle the case of no products in a price group
    if x == "[NONE]":
        return []
    return x.strip("[]").split(",")


def with_xywh(df: pd.DataFrame) -> pd.DataFrame:
    df["width"] = df["max_x"] - df["min_x"]
    df["height"] = df["max_y"] - df["min_y"]
    df["center_x"] = df["min_x"] + df["width"] / 2
    df["center_y"] = df["min_y"] + df["height"] / 2
    return df


def crop_from_xywhn(
    image: Image.Image,
    xywhn: list[float, float, float, float],
) -> Image.Image:
    W, H = image.size
    cx, cy, w, h = xywhn

    cx_abs = cx * W
    cy_abs = cy * H
    w_abs = w * W
    h_abs = h * H

    left = int(cx_abs - w_abs / 2)
    right = int(cx_abs + w_abs / 2)
    top = int(cy_abs - h_abs / 2)
    bottom = int(cy_abs + h_abs / 2)

    return image.crop((left, top, right, bottom))


@torch.inference_mode()
def get_bbox_and_embeddings_tensors(
    rows: pd.DataFrame,
    image: Image.Image,
    embedder: CLIP,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    xywh = torch.stack(
        [
            torch.tensor(rows["center_x"].values),
            torch.tensor(rows["center_y"].values),
            torch.tensor(rows["width"].values),
            torch.tensor(rows["height"].values),
        ],
        dim=1,
    ).to(torch.float32)

    embeddings = []
    for i in tqdm(
        range(0, len(xywh), batch_size), desc="Generating embeddings...", leave=False
    ):
        batch_bboxes = xywh[i : i + batch_size]
        batch_crops = [crop_from_xywhn(image, bbox) for bbox in batch_bboxes]
        batch_tensors = torch.stack(
            [preprocess(crop) for crop in batch_crops], dim=0
        ).to(device)
        batch_embeddings = embedder.encode_image(batch_tensors)
        batch_embeddings = F.normalize(batch_embeddings, dim=1)
        embeddings.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return xywh, embeddings


@torch.inference_mode()
def form_graph_components(
    scene_id: str,
    dataset_dir: Path,
    products_df: pd.DataFrame,
    price_tags_df: pd.DataFrame,
    groups_df: pd.DataFrame,
    embedder: CLIP,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> GraphComponents:
    """Form the components of a graph from a single scene.

    Args:
        scene_id (str): The ID of the scene to create a graph from.
        dataset_dir (Path): The directory containing the dataset.
        products_df (pd.DataFrame): Dataframe with product bounding boxes.
        price_tags_df (pd.DataFrame): Dataframe with price tag bounding boxes.
        groups_df (pd.DataFrame): Dataframe specifying price groups.
        embedder (CLIP): The embedder to use for generating node embeddings.
        preprocess (Callable[[Image.Image], torch.Tensor]): The preprocessing function to use for generating node embeddings.
        device (torch.device, optional): The device to use for CLIP inference. Defaults to torch.device("cpu").

    Returns:
        GraphComponents: An object containing the graph data, with the following fields:
            - "graph_id": The ID of the graph (its attributionset_id).
            - "global_embedding": A global embedding for the graph (represented as a tensor).
            - "product_bboxes": A mapping from product bbox IDs to their respective bounding boxes (represented as tensors).
            - "product_embeddings": A mapping from product bbox IDs to their respective embeddings (represented as tensors).
            - "price_bboxes": A mapping from price tag bbox IDs to their respective bounding boxes (represented as tensors).
            - "price_embeddings": A mapping from price tag bbox IDs to their respective embeddings (represented as tensors).
            - "upc_groups": A list of UPCGroup objects, where each object contains a UPC and a list of product bbox IDs that share that UPC.
            - "prod_price_groups": A list of ProductPriceGroup objects, where each object contains a group ID and a list of product and price tag bbox IDs that are in that pricing group.
    """
    # The double brackets in `loc` ensure we always return a dataframe, even if only one row matches
    scene_products: pd.DataFrame = products_df.loc[[scene_id]]
    scene_price_tags: pd.DataFrame = price_tags_df.loc[[scene_id]]
    scene_groups: pd.DataFrame = groups_df.loc[[scene_id]]
    group_ids_in_scene: list[str] = scene_groups["group_id"].unique().tolist()

    img_path = scene_products["local_path"].iloc[0]
    image = Image.open(dataset_dir / img_path)

    global_embedding = (
        embedder.encode_image(preprocess(image).to(device).unsqueeze(0)).cpu().flatten()
    )

    prod_bboxes, prod_embeddings = get_bbox_and_embeddings_tensors(
        rows=scene_products,
        image=image,
        embedder=embedder,
        preprocess=preprocess,
        device=device,
    )
    prod_bbox_ids = scene_products["prod_bbox_id"].tolist()
    price_bboxes, price_embeddings = get_bbox_and_embeddings_tensors(
        rows=scene_price_tags,
        image=image,
        embedder=embedder,
        preprocess=preprocess,
        device=device,
    )
    price_bbox_ids = scene_price_tags["price_bbox_id"].tolist()

    upc_groups = [
        UPCGroup(upc=upc, bbox_ids=bbox_ids)
        for upc, bbox_ids in scene_products.groupby("ml_label_name")["prod_bbox_id"]
        .apply(list)
        .items()
    ]

    # These groups provide the "ground truth" product-price edges.
    prod_price_groups = []
    for group_id in group_ids_in_scene:
        group_row = scene_groups[scene_groups["group_id"] == group_id].iloc[0]
        prod_bbox_ids_in_group = csv_array_str_to_list(group_row["product_bbox_ids"])
        price_bbox_ids_in_group = csv_array_str_to_list(group_row["price_bbox_ids"])
        prod_price_groups.append(
            ProductPriceGroup(
                group_id=group_id,
                product_bbox_ids=[
                    x for x in prod_bbox_ids_in_group if x in prod_bbox_ids
                ],
                price_bbox_ids=[
                    x for x in price_bbox_ids_in_group if x in price_bbox_ids
                ],
            )
        )

    return GraphComponents(
        graph_id=scene_id,
        global_embedding=global_embedding,
        product_bboxes=dict(zip(prod_bbox_ids, prod_bboxes)),
        product_embeddings=dict(zip(prod_bbox_ids, prod_embeddings)),
        price_bboxes=dict(zip(price_bbox_ids, price_bboxes)),
        price_embeddings=dict(zip(price_bbox_ids, price_embeddings)),
        upc_groups=upc_groups,
        prod_price_groups=prod_price_groups,
    )


def main(
    dataset_dir: Path,
    output_dir: Path,
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    device: torch.device = torch.device("cpu"),
    limit: int | None = None,
):
    assert train_pct + val_pct + test_pct == 1

    # Each dataframe is indexed by attribution set ID (indicates a single "price scene").
    products_df = pd.read_csv(
        dataset_dir / "product_boxes.csv", index_col=0, dtype={"ml_label_name": str}
    )
    price_tags_df = pd.read_csv(dataset_dir / "price_boxes.csv", index_col=0)
    groups_df = pd.read_csv(dataset_dir / "price_groups.csv", index_col=0)

    # Get scene IDs that we have data for (intersection of what's contained in all three tables).
    scene_ids_in_products_df = set(products_df.index.unique())
    scene_ids_in_price_tags_df = set(price_tags_df.index.unique())
    scene_ids_in_groups_df = set(groups_df.index.unique())
    scene_ids = list(
        scene_ids_in_products_df & scene_ids_in_price_tags_df & scene_ids_in_groups_df
    )
    if limit is not None:
        scene_ids = scene_ids[:limit]

    price_tags_df = with_xywh(price_tags_df)
    products_df = with_xywh(products_df)

    random.shuffle(scene_ids)
    num_scenes = len(scene_ids)
    train_cutoff = int(train_pct * num_scenes)
    val_cutoff = int((train_pct + val_pct) * num_scenes)
    trn_scene_ids = scene_ids[:train_cutoff]
    val_scene_ids = scene_ids[train_cutoff:val_cutoff]
    test_scene_ids = scene_ids[val_cutoff:]

    embedder, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )

    trn_components = []
    for scene_id in tqdm(trn_scene_ids, desc="Parsing training scenes into graphs..."):
        graph_components = form_graph_components(
            scene_id=scene_id,
            dataset_dir=dataset_dir,
            products_df=products_df,
            price_tags_df=price_tags_df,
            groups_df=groups_df,
            embedder=embedder,
            preprocess=preprocess,
            device=device,
        )
        trn_components.append(graph_components.model_dump())

    val_components = []
    for scene_id in tqdm(
        val_scene_ids, desc="Parsing validation scenes into graphs..."
    ):
        graph_components = form_graph_components(
            scene_id=scene_id,
            dataset_dir=dataset_dir,
            products_df=products_df,
            price_tags_df=price_tags_df,
            groups_df=groups_df,
            embedder=embedder,
            preprocess=preprocess,
            device=device,
        )
        val_components.append(graph_components.model_dump())

    test_components = []
    for scene_id in tqdm(test_scene_ids, desc="Parsing test scenes into graphs..."):
        graph_components = form_graph_components(
            scene_id=scene_id,
            dataset_dir=dataset_dir,
            products_df=products_df,
            price_tags_df=price_tags_df,
            groups_df=groups_df,
            embedder=embedder,
            preprocess=preprocess,
            device=device,
        )
        test_components.append(graph_components.model_dump())

    trn_graph_dir = output_dir / "train" / "raw"
    val_graph_dir = output_dir / "val" / "raw"
    test_graph_dir = output_dir / "test" / "raw"
    for d in trn_graph_dir, val_graph_dir, test_graph_dir:
        if not d.exists():
            d.mkdir(parents=True)
    torch.save(trn_components, trn_graph_dir / "graph_components.pt")
    torch.save(val_components, val_graph_dir / "graph_components.pt")
    torch.save(test_components, test_graph_dir / "graph_components.pt")

    products_df_trn: pd.DataFrame = products_df.loc[trn_scene_ids]
    products_df_val: pd.DataFrame = products_df.loc[val_scene_ids]
    products_df_test: pd.DataFrame = products_df.loc[test_scene_ids]
    price_tags_df_trn: pd.DataFrame = price_tags_df.loc[trn_scene_ids]
    price_tags_df_val: pd.DataFrame = price_tags_df.loc[val_scene_ids]
    price_tags_df_test: pd.DataFrame = price_tags_df.loc[test_scene_ids]

    for df in (
        products_df_trn,
        products_df_val,
        products_df_test,
        price_tags_df_trn,
        price_tags_df_val,
        price_tags_df_test,
    ):
        df.drop(columns=["width", "height", "center_x", "center_y"], inplace=True)

    products_df_trn.reset_index().to_csv(
        trn_graph_dir / "product_boxes.csv", index=False
    )
    products_df_val.reset_index().to_csv(
        val_graph_dir / "product_boxes.csv", index=False
    )
    products_df_test.reset_index().to_csv(
        test_graph_dir / "product_boxes.csv", index=False
    )
    price_tags_df_trn.reset_index().to_csv(
        trn_graph_dir / "price_boxes.csv", index=False
    )
    price_tags_df_val.reset_index().to_csv(
        val_graph_dir / "price_boxes.csv", index=False
    )
    price_tags_df_test.reset_index().to_csv(
        test_graph_dir / "price_boxes.csv", index=False
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-pct", type=float, default=0.8)
    parser.add_argument("--val-pct", type=float, default=0.1)
    parser.add_argument("--test-pct", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        device=torch.device(args.device),
        limit=args.limit,
    )
