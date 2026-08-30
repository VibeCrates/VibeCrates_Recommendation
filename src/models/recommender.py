"""
Recommender model implementations based on the user's architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPVisionModel, CLIPTextModel, CLIPConfig
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
    def __init__(self, model_name: str = 'all-mpnet-base-v2', output_dim: int = 768):
        super().__init__()
        self.sbert = SentenceTransformer(model_name)
        for param in self.sbert.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],  # all-mpnet-base-v2 uses q/v not query/value
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        # Replace the underlying transformer with its LoRA-wrapped version so that
        # self.sbert.encode() actually runs through the LoRA adapter during forward.
        transformer_model = self.sbert._first_module().auto_model
        self.sbert._first_module().auto_model = get_peft_model(transformer_model, lora_config)

        self.mlp = MLP(self.sbert.get_sentence_embedding_dimension(), output_dim)

    def forward(self, text_list: list[str]):
        transformer_module = self.sbert._first_module()
        device = next(self.parameters()).device
        tokenized = transformer_module.tokenizer(
            text_list, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        model_output = transformer_module.auto_model(**tokenized)
        # Mean pooling over non-padding tokens — mirrors sentence-transformers default
        token_embeddings = model_output.last_hidden_state  # (B, T, H)
        mask = tokenized["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.mlp(embeddings)


class ImageBlock(nn.Module):
    """
    Encodes an image using CLIP's vision encoder and an MLP.
    Input: content_image (224, 224, 3)
    Output: z_image (768 dim)
    """
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14', output_dim: int = 768):
        super().__init__()
        # 이미지 프로세서는 **원본 이미지가 실제로 들어올 때만** 만든다(아래 processor 속성).
        # 서빙에서는 미리 뽑아 둔 CLIP 특징(B, 1024)만 들어오므로 이 경로를 아예 타지 않고,
        # 그 환경에는 PIL이 없어도 된다. 미리 만들면 모델 생성 자체가 실패한다.
        self._model_name = model_name
        self._processor = None
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.mlp = MLP(self.vision_encoder.config.hidden_size, output_dim)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = CLIPImageProcessor.from_pretrained(self._model_name)
        return self._processor

    def forward(self, images):
        if isinstance(images, torch.Tensor):
            # Pre-computed CLIP features (B, 1024) — skip frozen encoder
            return self.mlp(images.to(next(self.mlp.parameters()).device))
        inputs = self.processor(images=images, return_tensors="pt").to(self.vision_encoder.device)
        image_features = self.vision_encoder(**inputs).pooler_output
        return self.mlp(image_features)


class QueryBlock(nn.Module):
    """
    Encodes a query text using CLIP's text encoder and an MLP.
    Input: query_text (max 77 tokens)
    Output: z_query (768 dim)
    """
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14', output_dim: int = 768,
                 use_lora: bool = False):
        super().__init__()
        # 텍스트만 쓰는데도 CLIPProcessor를 쓰면 이미지 프로세서까지 함께 만들어진다.
        # 현재 transformers 버전은 이 체크포인트의 image processor를 인식하지 못해
        # ("Unrecognized image processor") **모델 객체 생성 자체가 실패한다**.
        # CLIPTokenizer는 그 경로를 아예 거치지 않고, 토큰화 결과는 동일하다(대조 확인함).
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        # Freeze CLIP parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 개선안 1 (설계 노트 근본원인 1) — 쿼리 인코더 대칭화.
        # 콘텐츠 쪽 TextBlock은 SBERT+LoRA로 적응 가능한데 쿼리 쪽만 완전 동결이라
        # 인코더가 비대칭이었다. CLIP text는 이미지-캡션으로 학습돼 구체·시각 언어에
        # 강하고 은유에 약한데, 동결이라 poet 언어에 적응할 capacity가 없다.
        # LoRA를 붙여 소량의 파라미터만 여는 것이 TextBlock과 대칭인 최소 변경안이다.
        # target_modules는 CLIP의 어텐션 명명(q_proj/v_proj)을 따른다 — mpnet의 q/v와 다르다.
        self.use_lora = use_lora
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                # task_type은 지정하지 않는다. FEATURE_EXTRACTION을 주면 peft가
                # PeftModelForFeatureExtraction으로 감싸는데, 그 forward가 inputs_embeds를
                # 넘겨 CLIPTextTransformer 내부의 encoder 호출과 충돌한다
                # ("got multiple values for keyword argument 'inputs_embeds'").
                # task_type 없이 감싸면 forward가 base 모델로 그대로 위임된다.
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        self.mlp = MLP(self.text_encoder.config.hidden_size, output_dim)

    def forward(self, queries: list[str]):
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.text_encoder.device)
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


class DualEncoderModel(nn.Module, BaseRecommender):
    """
    The main model that orchestrates all blocks for training and inference.
    """
    def __init__(self, query_lora: bool = False):
        nn.Module.__init__(self)
        BaseRecommender.__init__(self, name="DualEncoderModel")
        self.text_block = TextBlock()
        self.image_block = ImageBlock()
        # query_lora=True면 QueryBlock의 CLIP 텍스트 인코더에 LoRA가 붙는다(개선안 1).
        # 그 파라미터 이름에 'lora'가 들어가므로 Stage 1의 requires_grad 규칙
        # ('lora' or 'mlp')에 자동으로 포함된다 — 트레이너 수정은 필요 없다.
        self.query_block = QueryBlock(use_lora=query_lora)
        self.content_block = ContentBlock()

    def encode_content(self, text_list: list[str], images):
        """Encodes content from text and image."""
        z_text = self.text_block(text_list)
        z_image = self.image_block(images)
        z_content = self.content_block(z_image, z_text)
        return z_content, z_text, z_image

    def encode_query(self, queries: list[list[str]] | list[str]) -> torch.Tensor:
        """
        Encodes N queries per item via QueryBlock then mean-pools → z_query (B, 768).
        Accepts both List[str] (single-query inference) and List[List[str]] (training, DSV).
        Items with no queries get a zero vector.
        """
        if queries and isinstance(queries[0], str):
            queries = [[q] for q in queries]

        flat = [q for qs in queries for q in qs]
        counts = [len(qs) for qs in queries]
        out_dim = self.query_block.mlp.layers[-1].out_features
        device = next(self.parameters()).device

        if not flat:
            return torch.zeros(len(queries), out_dim, device=device)

        z_flat = self.query_block(flat)  # (sum(N), D)

        pooled = []
        offset = 0
        for n in counts:
            if n == 0:
                pooled.append(torch.zeros(out_dim, device=device))
            else:
                pooled.append(z_flat[offset:offset + n].mean(dim=0))
            offset += n

        return torch.stack(pooled)  # (B, D)

    def forward(self, batch):
        """
        batch 형식: {'content_text': List[str], 'content_image': List[PIL],
                     'query': List[List[str]]}
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
