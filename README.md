# MLOps – End-to-End Production-Ready ML System

## Project Overview

This project implements a simplified end-to-end MLOps architecture designed to simulate a real-world production environment.

The system is structured around three independent components:

1. Model Training  
2. Experiment Tracking & Model Registry  
3. Model Serving API  

All services are containerized and orchestrated using Docker Compose to ensure reproducibility and environment consistency.

---

## Architecture

```
Training Script
    ↓
MLflow Tracking Server
    ↓
Model Registry
    ↓
FastAPI Service loads Production model
    ↓
Client sends /predict request
    ↓
JSON response
```

This separation reflects common production practices where training, tracking, and serving are independent services.

---

## Project Structure

```
mlops-starter/
│
├── docker-compose.yml
├── .env
├── services/
│   ├── mlflow/
│   │   └── Dockerfile
│   │
│   └── api/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── main.py
│           └── schemas.py
│
└── training/
    ├── requirements.txt
    └── train.py
```

---

## Component Breakdown

### docker-compose.yml

Orchestrates the entire system.

**Responsibilities:**

- Defines services (MLflow + API)
- Configures networking between containers
- Maps ports to the host machine
- Manages persistent volumes
- Injects environment variables

Ensures the full stack can be reproduced with a single command.

---

### .env

Stores environment-specific configuration such as:

- Service ports
- Model name
- Model stage

Avoids hardcoded configuration and follows best practices for environment management.

---

### training/

Purpose: Model experimentation and registration.

**Responsibilities:**

- Load dataset
- Train model
- Log parameters and metrics
- Register model in MLflow Model Registry

This module is strictly separated from serving logic to maintain clean architectural boundaries.

---

### services/mlflow/

Runs the MLflow Tracking Server.

**Responsibilities:**

- Store experiment metadata
- Store metrics and parameters
- Manage model artifacts
- Provide Model Registry functionality

Acts as the backbone for experiment tracking and model versioning.

---

### services/api/

Exposes the trained model through a REST API.

**Responsibilities:**

- Load model from MLflow Registry
- Provide prediction endpoint (`/predict`)
- Provide health check endpoint (`/health`)
- Validate requests using Pydantic schemas

The API is fully containerized and isolated from training logic.

---

## Design Principles

This project follows key MLOps and production engineering principles:

- Separation of concerns (training, tracking, serving)
- Containerized reproducibility
- Environment configuration isolation
- Model versioning via registry
- API-based model serving
- Minimal but production-oriented architecture

---

## How to Run

Build and start all services:

```bash
docker compose up --build
```

**Services:**

- MLflow UI → http://localhost:5000  
- API → http://localhost:8000  

---

## Objective

The goal of this project is to demonstrate the ability to:

- Train and version ML models
- Containerize services
- Build production-ready APIs
- Implement experiment tracking workflows
- Structure an ML system using reproducible architecture
