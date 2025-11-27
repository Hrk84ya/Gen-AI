# 🚀 Production Deployment

## 🎯 Learning Objectives
- Deploy AI models to production environments
- Implement MLOps best practices
- Monitor and maintain deployed models
- Scale AI applications effectively

## 🏗️ Deployment Architecture

### Model Serving Options
```python
# 1. REST API with Flask
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```python
# 2. FastAPI (Recommended)
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )
```

## 🐳 Containerization with Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pkl
    volumes:
      - ./models:/app/models
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

## ☁️ Cloud Deployment

### AWS Deployment
```python
# Using AWS SageMaker
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Create SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Deploy model
sklearn_estimator = SKLearn(
    entry_point='inference.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3'
)

predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

### Google Cloud Platform
```python
# Using Google Cloud AI Platform
from google.cloud import aiplatform

aiplatform.init(project='your-project-id', location='us-central1')

# Deploy model
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://your-bucket/model/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-23:latest'
)

endpoint = model.deploy(
    machine_type='n1-standard-2',
    min_replica_count=1,
    max_replica_count=3
)
```

### Azure ML
```python
# Using Azure Machine Learning
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice

ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path='model.pkl',
    model_name='my-model'
)

# Deploy to Azure Container Instances
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    description='My ML Model'
)

service = Model.deploy(
    workspace=ws,
    name='my-model-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)
```

## 📊 Model Monitoring

### Performance Monitoring
```python
import logging
from datetime import datetime
import json

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.logger = logging.getLogger(f'{model_name}_monitor')
    
    def log_prediction(self, input_data, prediction, confidence, latency):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': self.model_name,
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency,
            'input_size': len(input_data) if isinstance(input_data, list) else 1
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_type, error_message, input_data=None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': self.model_name,
            'error_type': error_type,
            'error_message': error_message,
            'input_data': str(input_data) if input_data else None
        }
        
        self.logger.error(json.dumps(log_entry))

# Usage in API
monitor = ModelMonitor('my-classifier')

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    try:
        prediction = model.predict([request.features])
        confidence = model.predict_proba([request.features]).max()
        latency = (time.time() - start_time) * 1000
        
        monitor.log_prediction(
            request.features, prediction[0], confidence, latency
        )
        
        return {"prediction": prediction[0], "confidence": confidence}
    
    except Exception as e:
        monitor.log_error("prediction_error", str(e), request.features)
        raise HTTPException(status_code=500, detail="Prediction failed")
```

### Data Drift Detection
```python
import numpy as np
from scipy import stats
from sklearn.metrics import jensen_shannon_distance

class DriftDetector:
    def __init__(self, reference_data, threshold=0.1):
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._calculate_stats(reference_data)
    
    def _calculate_stats(self, data):
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'distribution': np.histogram(data.flatten(), bins=50)[0]
        }
    
    def detect_drift(self, new_data):
        new_stats = self._calculate_stats(new_data)
        
        # Statistical tests
        drift_scores = {}
        
        # KS test for distribution drift
        ks_stat, ks_p = stats.ks_2samp(
            self.reference_data.flatten(),
            new_data.flatten()
        )
        drift_scores['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}
        
        # Jensen-Shannon divergence
        js_distance = jensen_shannon_distance(
            self.reference_stats['distribution'],
            new_stats['distribution']
        )
        drift_scores['js_divergence'] = js_distance
        
        # Determine if drift occurred
        drift_detected = (
            ks_p < 0.05 or  # Significant difference in distributions
            js_distance > self.threshold  # High divergence
        )
        
        return {
            'drift_detected': drift_detected,
            'scores': drift_scores,
            'timestamp': datetime.utcnow().isoformat()
        }
```

## 🔄 MLOps Pipeline

### CI/CD for ML Models
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/
    
    - name: Train model
      run: python train.py
    
    - name: Validate model
      run: python validate_model.py
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to staging
      run: |
        docker build -t my-model:latest .
        docker push my-registry/my-model:latest
    
    - name: Run integration tests
      run: python integration_tests.py
    
    - name: Deploy to production
      run: kubectl apply -f k8s/deployment.yaml
```

### Model Versioning
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.register_model(model_uri, "MyClassifier")

# Load specific model version
client = MlflowClient()
model_version = client.get_latest_versions("MyClassifier", stages=["Production"])[0]
model = mlflow.sklearn.load_model(f"models:/{model_version.name}/{model_version.version}")
```

## ⚡ Performance Optimization

### Model Optimization
```python
# 1. Model Quantization (PyTorch)
import torch.quantization as quantization

# Post-training quantization
model_fp32 = MyModel()
model_fp32.eval()

# Quantize model
model_int8 = quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# 2. ONNX Conversion
import torch.onnx
import onnxruntime

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# Load with ONNX Runtime
ort_session = onnxruntime.InferenceSession("model.onnx")

def predict_onnx(input_data):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs[0]
```

### Caching and Load Balancing
```python
import redis
import pickle
import hashlib

class ModelCache:
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.ttl = ttl
    
    def _get_cache_key(self, input_data):
        input_str = str(sorted(input_data.items()) if isinstance(input_data, dict) else input_data)
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def get_prediction(self, input_data):
        cache_key = self._get_cache_key(input_data)
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return pickle.loads(cached_result)
        return None
    
    def cache_prediction(self, input_data, prediction):
        cache_key = self._get_cache_key(input_data)
        self.redis_client.setex(
            cache_key, 
            self.ttl, 
            pickle.dumps(prediction)
        )

# Usage in API
cache = ModelCache()

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Check cache first
    cached_result = cache.get_prediction(request.features)
    if cached_result:
        return cached_result
    
    # Make prediction
    prediction = model.predict([request.features])
    result = {"prediction": prediction[0]}
    
    # Cache result
    cache.cache_prediction(request.features, result)
    
    return result
```

## 🔒 Security Best Practices

### API Security
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

app = FastAPI()
security = HTTPBearer()

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/predict")
async def predict(request: PredictionRequest, username: str = Depends(verify_token)):
    # Your prediction logic here
    pass
```

### Input Validation
```python
from pydantic import BaseModel, validator
import numpy as np

class PredictionRequest(BaseModel):
    features: list
    
    @validator('features')
    def validate_features(cls, v):
        if not isinstance(v, list):
            raise ValueError('Features must be a list')
        
        if len(v) != 10:  # Expected number of features
            raise ValueError('Features must contain exactly 10 values')
        
        # Check for valid numeric values
        try:
            numeric_features = [float(x) for x in v]
        except (ValueError, TypeError):
            raise ValueError('All features must be numeric')
        
        # Check for reasonable ranges
        if any(abs(x) > 1000 for x in numeric_features):
            raise ValueError('Feature values seem unreasonable')
        
        return numeric_features
```

## 📈 Scaling Strategies

### Horizontal Scaling with Kubernetes
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: api
        image: my-registry/ml-model:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: MODEL_PATH
          value: "/app/model.pkl"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Batch Processing
```python
import asyncio
from typing import List
import time

class BatchProcessor:
    def __init__(self, model, batch_size=32, max_wait_time=0.1):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False
    
    async def predict(self, features):
        # Add request to batch
        future = asyncio.Future()
        self.pending_requests.append((features, future))
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        self.processing = True
        
        while self.pending_requests:
            # Wait for batch to fill or timeout
            start_time = time.time()
            while (len(self.pending_requests) < self.batch_size and 
                   time.time() - start_time < self.max_wait_time):
                await asyncio.sleep(0.01)
            
            # Process current batch
            if self.pending_requests:
                batch = self.pending_requests[:self.batch_size]
                self.pending_requests = self.pending_requests[self.batch_size:]
                
                # Extract features and futures
                features_batch = [item[0] for item in batch]
                futures = [item[1] for item in batch]
                
                # Make batch prediction
                try:
                    predictions = self.model.predict(features_batch)
                    for future, prediction in zip(futures, predictions):
                        future.set_result(prediction)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
        
        self.processing = False

# Usage
batch_processor = BatchProcessor(model)

@app.post("/predict")
async def predict(request: PredictionRequest):
    prediction = await batch_processor.predict(request.features)
    return {"prediction": prediction}
```

## 🎯 Deployment Checklist

### Pre-deployment
- [ ] Model performance validated on test data
- [ ] Code reviewed and tested
- [ ] Security measures implemented
- [ ] Monitoring and logging configured
- [ ] Error handling implemented
- [ ] Documentation updated

### Deployment
- [ ] Staging environment tested
- [ ] Load testing completed
- [ ] Rollback plan prepared
- [ ] Health checks configured
- [ ] Alerts set up
- [ ] Team notified

### Post-deployment
- [ ] Monitor initial performance
- [ ] Check error rates and latency
- [ ] Validate predictions on real data
- [ ] Monitor resource usage
- [ ] Collect user feedback
- [ ] Plan for model updates

---
*Continue to [Latest Research](../06-Research/) →*