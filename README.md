# 🚀 Nine Thousand KFP

🤖 **ML Pipeline Powerhouse** - A comprehensive Kubeflow Pipelines (KFP) implementation for end-to-end machine learning workflows.

## 🎯 Purpose

This repository provides a production-ready ML training pipeline that:

- 📊 **Fetches Data** from multiple sources (GitHub, DVC, Feast)
- ✅ **Validates** incoming datasets for quality assurance  
- 🔧 **Preprocesses** data for optimal model training
- 🧠 **Trains** Keras models with configurable hyperparameters
- 🔄 **Converts** models to ONNX format for deployment
- 📈 **Evaluates** model performance with comprehensive metrics
- 🏪 **Registers** trained models to a model registry

## 🏗️ Architecture

```
Data Sources → Validation → Preprocessing → Training → Conversion → Evaluation → Registry
```

## 🚀 Quick Start

1. Configure your pipeline in `parts-pipeline/pipeline_config.yaml`
2. Run the training pipeline: `python prod_train_save_pipeline.py`
3. Monitor your ML workflow through Kubeflow UI

Perfect for MLOps teams building scalable, reproducible machine learning pipelines! 🎉