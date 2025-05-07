import torch
from pydantic import BaseModel
from pydantic import model_validator


class UPCGroup(BaseModel):
    upc: str
    bbox_ids: list[str]


class ProductPriceGroup(BaseModel):
    group_id: str
    product_bbox_ids: list[str]
    price_bbox_ids: list[str]


class UndirectedGraph(BaseModel):
    id_to_idx: dict[str, int]
    x: torch.Tensor
    edge_index: torch.LongTensor
    edge_attr: torch.Tensor
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_graph_is_undirected(self):
        edge_set = set(map(tuple, self.edge_index.T.tolist()))
        for i, j in edge_set:
            if (j, i) not in edge_set:
                raise ValueError(
                    f"Graph should be undirected, but edge ({i}, {j}) is present without ({j}, {i})."
                )
        return self

    @model_validator(mode="after")
    def check_no_self_loops(self):
        edge_set = set(map(tuple, self.edge_index.T.tolist()))
        for i, j in edge_set:
            if i == j:
                raise ValueError(
                    f"Graph should have no self loops, but edge ({i}, {j}) is present."
                )
        return self

    @model_validator(mode="after")
    def check_edge_index_refers_to_x(self):
        if torch.any(self.edge_index >= self.x.shape[0]):
            raise ValueError("Edge index must refer to valid indices in `x`.")
        return self

    @model_validator(mode="after")
    def check_edge_attr_shape(self):
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError(
                "Edge attributes must have the same number of edges as the edge index."
            )
        return self

    @model_validator(mode="after")
    def check_id_to_idx(self):
        if len(self.id_to_idx) != len(self.x):
            raise ValueError(
                "ID to index mapping must have the same number of IDs as nodes in `x`."
            )
        for bbox_id in self.id_to_idx:
            if self.id_to_idx[bbox_id] >= self.x.shape[0]:
                raise ValueError(
                    f"ID to index mapping specifies an invalid index ({self.id_to_idx[bbox_id]}) for bbox ID {bbox_id}."
                )
        return self


class GraphComponents(BaseModel):
    graph_id: str
    global_embedding: torch.Tensor
    product_bboxes: dict[str, torch.Tensor]
    product_embeddings: dict[str, torch.Tensor]
    price_bboxes: dict[str, torch.Tensor]
    price_embeddings: dict[str, torch.Tensor]
    upc_groups: list[UPCGroup]
    prod_price_groups: list[ProductPriceGroup]
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def check_upc_groups_contain_product_ids_only(self):
        for group in self.upc_groups:
            for bbox_id in group.bbox_ids:
                if bbox_id not in self.product_bboxes:
                    raise ValueError(
                        f"UPC group {group} contains product bbox ID {bbox_id} that is not in the graph."
                    )
        return self

    @model_validator(mode="after")
    def check_prod_price_groups_contain_product_and_price_ids_only(self):
        for group in self.prod_price_groups:
            for bbox_id in group.product_bbox_ids:
                if bbox_id not in self.product_bboxes:
                    raise ValueError(
                        f"Product-price group {group} contains product bbox ID {bbox_id} that is not in the graph."
                    )
            for bbox_id in group.price_bbox_ids:
                if bbox_id not in self.price_bboxes:
                    raise ValueError(
                        f"Product-price group {group} contains price bbox ID {bbox_id} that is not in the graph."
                    )
        return self

    @model_validator(mode="after")
    def check_embeddings_are_all_flat_and_same_dim(self):
        if self.global_embedding.ndim != 1:
            raise ValueError("Global embedding must be a flattened tensor.")
        embedding_dim = len(self.global_embedding)
        for bbox_id, embedding in self.product_embeddings.items():
            if embedding.ndim != 1 or embedding.shape[0] != embedding_dim:
                raise ValueError(
                    f"Product embedding for bbox ID {bbox_id} has different dimension than global embedding."
                )
        for bbox_id, embedding in self.price_embeddings.items():
            if embedding.ndim != 1 or embedding.shape[0] != embedding_dim:
                raise ValueError(
                    f"Price embedding for bbox ID {bbox_id} has different dimension than global embedding."
                )
        return self
