"""
Custom loss functions for training the recommendation model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for contrastive learning.
    This is a generic implementation for two sets of embeddings.
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: A hyperparameter to scale the logits.
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, key_embeddings: torch.Tensor):
        """
        Calculates the symmetric InfoNCE loss between query and key embeddings.

        Args:
            query_embeddings: A tensor of shape (N, D)
            key_embeddings: A tensor of shape (N, D)

        Returns:
            The calculated InfoNCE loss.
        """
        # Normalize embeddings to have unit length
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        key_embeddings = F.normalize(key_embeddings, p=2, dim=1)

        # Calculate cosine similarity matrix
        # The logits are the scaled cosine similarities
        logits = torch.matmul(query_embeddings, key_embeddings.T) / self.temperature

        # Create labels for cross-entropy loss.
        # For a batch of size N, the positive pair for the i-th query is the i-th key.
        # So the labels are just [0, 1, 2, ..., N-1].
        batch_size = query_embeddings.shape[0]
        labels = torch.arange(batch_size, device=query_embeddings.device)

        # Calculate loss in both directions (q->k and k->q)
        loss_q_k = self.criterion(logits, labels)
        loss_k_q = self.criterion(logits.T, labels)

        # The final loss is the average of the two
        loss = (loss_q_k + loss_k_q) / 2.0
        return loss

class KLDivergenceLoss(nn.Module):
    """
    Calculates the KL-Divergence loss for knowledge distillation.
    It measures the difference between two probability distributions.
    """
    def __init__(self):
        super().__init__()
        # Using log_target=True because we will pass log-probabilities
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, z_content_dist: torch.Tensor, z_query_dist: torch.Tensor):
        """
        Calculates the KL-Divergence between the content and query distributions.

        Args:
            z_content_dist: Log-softmax probabilities of the content vectors. Shape (N, D)
            z_query_dist: Log-softmax probabilities of the query vectors (the "teacher"). Shape (N, D)

        Returns:
            The calculated KL-Divergence loss.
        """
        # The KLDivLoss expects the input (student) to be log-probabilities
        # and the target (teacher) can also be log-probabilities if log_target=True.
        return self.kl_div(z_content_dist, z_query_dist)
