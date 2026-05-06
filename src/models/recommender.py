"""
Recommender model implementations based on the user's architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTextModel, CLIPConfig
from peft import get_peft_model, LoraConfig, TaskType

from .base import BaseRecommender


class MLP(nn.Module):
    """A simple MLP block."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class TextBlock(nn.Module):
    """
    Encodes content text using SBERT with LoRA and an MLP.
    Input: content_text (max 512 tokens)
    Output: z_text (768 dim)
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2', output_dim: int = 768):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        # Freeze SBERT parameters
        for param in self.sbert.parameters():
            param.requires_grad = False
        
        # Apply LoRA to the SBERT model
        # Note: Applying LoRA to SentenceTransformer requires accessing the underlying transformer model.
        # This is a simplified approach. For a full implementation, you might need to adapt the SentenceTransformer's modules.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"], # Common target modules in transformers
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        # Get the underlying transformer model to apply PEFT
        transformer_model = self.sbert._first_module().auto_model
        self.sbert_lora = get_peft_model(transformer_model, lora_config)

        self.mlp = MLP(self.sbert.get_sentence_embedding_dimension(), output_dim)

    def forward(self, text_list: list[str]):
        # SentenceTransformer expects a list of strings
        # The model will handle tokenization, padding, and truncation internally.
        embeddings = self.sbert.encode(text_list, convert_to_tensor=True, device=self.sbert.device)
        z_text = self.mlp(embeddings)
        return z_text


class ImageBlock(nn.Module):
    """
    Encodes an image using CLIP's vision encoder and an MLP.
    Input: content_image (224, 224, 3)
    Output: z_image (768 dim)
    """
    def __init__(self, model_name: str = 'Bingsu/clip-vit-large-patch14-ko', output_dim: int = 768):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.mlp = MLP(self.vision_encoder.config.hidden_size, output_dim)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.vision_encoder.device)
        image_features = self.vision_encoder(**inputs).pooler_output
        z_image = self.mlp(image_features)
        return z_image


class QueryBlock(nn.Module):
    """
    Encodes a query text using CLIP's text encoder and an MLP.
    Input: query_text (max 77 tokens)
    Output: z_query (768 dim)
    """
    def __init__(self, model_name: str = 'Bingsu/clip-vit-large-patch14-ko', output_dim: int = 768):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        # Freeze CLIP parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.mlp = MLP(self.text_encoder.config.hidden_size, output_dim)

    def forward(self, queries: list[str]):
        inputs = self.processor(text=queries, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.text_encoder.device)
        text_features = self.text_encoder(**inputs).pooler_output
        z_query = self.mlp(text_features)
        return z_query


class ContentBlock(nn.Module):
    """
    Creates the final content vector by combining image and text vectors.
    Input: z_image (768 dim), z_text (768 dim)
    Output: z_content (768 dim)
    """
    def __init__(self, input_dim: int = 1536, output_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim)
        )

    def forward(self, z_image, z_text):
        combined = torch.cat((z_image, z_text), dim=1)
        z_content = self.layers(combined)
        z_content = F.normalize(z_content, p=2, dim=1)
        return z_content


class DualEncoderModel(BaseRecommender):
    """
    The main model that orchestrates all blocks for training and inference.
    """
    def __init__(self):
        super().__init__(name="DualEncoderModel")
        self.text_block = TextBlock()
        self.image_block = ImageBlock()
        self.query_block = QueryBlock()
        self.content_block = ContentBlock()

    def encode_content(self, text_list: list[str], images):
        """Encodes content from text and image."""
        z_text = self.text_block(text_list)
        z_image = self.image_block(images)
        z_content = self.content_block(z_image, z_text)
        return z_content, z_text, z_image

    def encode_query(self, queries: list[str]):
        """Encodes a search query."""
        z_query = self.query_block(queries)
        return z_query

    def forward(self, batch):
        """
        A versatile forward pass for training.
        The batch should be a dictionary containing the necessary data.
        e.g., {'content_text': [...], 'content_image': ..., 'query': [...]}
        """
        z_content, z_text, z_image = self.encode_content(batch['content_text'], batch['content_image'])
        z_query = self.encode_query(batch['query'])
        
        return {
            "z_content": z_content,
            "z_text": z_text,
            "z_image": z_image,
            "z_query": z_query
        }

    def fit(self, X, y=None, **kwargs):
        """Training logic will be handled by the Trainer class."""
        raise NotImplementedError("Use the custom Trainer class for training this model.")

    def predict(self, X, top_k=10):
        """Prediction logic will be handled by an inference script with ANN search."""
        raise NotImplementedError("Use a dedicated inference script for predictions.")

    def evaluate(self, X_test, y_test=None, **kwargs):
        """Evaluation logic will be handled by the Trainer or a dedicated script."""
        raise NotImplementedError("Use the custom Trainer for evaluation.")
