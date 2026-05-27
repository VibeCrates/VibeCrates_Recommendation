# VibeCrates Recommendation System

A recommendation system that provides personalized recommendations to users.

## Project Structure

```
VibeCrates_Recommendation/
├── data/                    # Training and test data
│   ├── raw/                # Raw data
│   └── processed/          # Preprocessed data
├── models/                 # Trained model parameters
│   ├── checkpoints/        # Checkpoints during training
│   └── final/              # Final trained models
├── src/                    # Source code
│   ├── models/             # Recommendation model implementations
│   ├── data/               # Data loading and preprocessing
│   ├── training/           # Model training code
│   └── api/                # FastAPI server implementation
├── scripts/                # Training, evaluation, and inference scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── docker/                 # Docker configuration
└── requirements.txt        # Python dependencies
```

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd VibeCrates_Recommendation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

- Add raw data to `data/raw/` directory
- Expected data structure:
  - `users.csv`: User information
  - `items.csv`: Item information
  - `interactions.csv`: User-item interaction data

## Usage

### Train Model

```bash
python scripts/train.py \
    --model_type collaborative_filtering \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 0.001
```

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model_path models/final/model.pkl
```

### Single User Inference

```bash
python scripts/inference.py \
    --user_id 123 \
    --top_k 10
```

### Run API Server

```bash
# Development server with auto-reload
python -m uvicorn src.api.main:app --reload --port 8000

# Production server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API documentation is available at http://localhost:8000/docs

### Run with Docker

```bash
# Build image
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

### Generate Recommendations
```
POST /api/v1/recommend
Content-Type: application/json

{
    "user_id": 123,
    "num_recommendations": 10,
    "filters": {}
}
```

Or

```
GET /api/v1/recommend/{user_id}?top_k=10
```

### Get User Profile
```
GET /api/v1/user-profile/{user_id}
```

### Get Item Information
```
GET /api/v1/item-info/{item_id}
```

### Submit Feedback
```
POST /api/v1/feedback?user_id=123&item_id=456&rating=4.5
```

## Key Files

### Model Implementation
- [src/models/base.py](src/models/base.py): Base recommender abstract class
- [src/models/recommender.py](src/models/recommender.py): Various recommendation model implementations
- [src/models/utils.py](src/models/utils.py): Model utility functions

### Data Processing
- [src/data/loader.py](src/data/loader.py): Data loading
- [src/data/preprocessing.py](src/data/preprocessing.py): Data preprocessing and feature engineering

### Training
- [src/training/trainer.py](src/training/trainer.py): Model training loop
- [src/training/config.py](src/training/config.py): Training configuration and hyperparameters

### API
- [src/api/main.py](src/api/main.py): FastAPI app main
- [src/api/routes.py](src/api/routes.py): API endpoints
- [src/api/schemas.py](src/api/schemas.py): Request/response schemas
- [src/api/dependencies.py](src/api/dependencies.py): Dependency injection (model loading, etc.)

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Verbose output
pytest tests/ -v
```

## Development Guide

### Add New Model

1. Inherit from `BaseRecommender` in `src/models/recommender.py`
2. Implement required methods: `fit()`, `predict()`, `evaluate()`
3. Configure model in `scripts/train.py`

### Add API Endpoint

1. Define request/response schemas in `src/api/schemas.py`
2. Implement route function in `src/api/routes.py`
3. Write tests in `tests/test_api.py`

### Add Preprocessing Logic

1. Add methods to `DataPreprocessor` class in `src/data/preprocessing.py`
2. Apply preprocessing in `scripts/train.py`

## Performance Optimization

- Model Caching: Load model at server startup (warm start)
- Batch Processing: Batch process large prediction requests
- Monitoring: Monitor prediction latency and accuracy

## Known Issues

TODO: To be added

## License

MIT License

## Contact

email address : maxsky9018@gmail.com
