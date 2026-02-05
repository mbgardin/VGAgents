import torch
import torch.nn as nn

class BucketedEmbedding(nn.Embedding):
    """
    Expects: BucketedEmbedding(bucket_size, n_values, embedding_dim, padding_idx=0)
    where n_values = max_value + 1 for the discrete variable BEFORE bucketing.
    """
    def __init__(self, bucket_size: int, n_values: int, embedding_dim: int, padding_idx: int = 0):
        self.bucket_size = int(bucket_size)
        self.n_values = int(n_values)

        # After bucketing, indices range is [0, floor((n_values-1)/bucket_size)]
        num_embeddings = (self.n_values - 1) // self.bucket_size + 1

        # Ensure padding_idx is valid
        if padding_idx is None:
            padding_idx = 0
        padding_idx = int(padding_idx)
        if padding_idx < 0 or padding_idx >= num_embeddings:
            padding_idx = 0

        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=int(embedding_dim),
                         padding_idx=padding_idx)

    def forward(self, indices):
        # integer bucketization -> valid embedding indices
        bucketed = torch.div(indices, self.bucket_size, rounding_mode="floor").long()
        # clamp just in case upstream gives out-of-range values
        bucketed = bucketed.clamp(min=0, max=self.num_embeddings - 1)
        return super().forward(bucketed)