# DeepLearning-Image-and-Text
Developed and trained deep learning models including LeNet-5 CNN for image classification and nanoGPT/DistilGPT2 for generative text modeling, exploring AI architecture design, hyperparameter tuning, and inference using PyTorch and HuggingFace.

This project explores deep learning through three applied tasks:
1. Implementing and training LeNet-5 Convolutional Neural Network (CNN) on the CIFAR-100 image classification dataset.
2. Training nanoGPT, a lightweight character-level GPT model, on Shakespeare text using PyTorch.
3. Fine-tuning DistilGPT2 (HuggingFace Transformers) on custom text data for controlled text generation.

# Project Objectives 
- Build and train deep learning models using PyTorch.
- Evaluate performance under different hyperparameters (batch size, learning rate, epochs).
- Implement Generative AI models with sampling strategies (temperature, top-k, top-p).
- Analyze trainable parameters, efficiency, and model behavior.

# Project Structure 
LeNet: 
- student_code.py: Model implementation
- train_cifar100.py: Training script
- eval_cifar100.py: Evaluation script
- results.txt: Recorded validation accuracy
- outputs/: Stored model checkpoints

nanoGPT/
- train.py: Training script
- sample.py: Text generation
- generated_nanogpt.txt: Generated Shakespeare-style text

DistilGPT2/
- distilgpt2_sft_cpu.py: Fine-tuning and generation script
- data.csv: Custom training dataset
- generated_distilgpt2.txt: Generated text samples

README.md: Project description and documentation 

# Part 1: LeNet CNN on CIFAR-100
Features: 
- Built CNN architecture from scratch using PyTorch
- Layer-by-layer shape monitoring (shape_dict)
- Counted trainable parameters manually
- Tested with different hyperparameter configurations

# Part 2: NanoGPT - Shakespeare Text Generation
Goals: 
- Train a minimal GPT model on Shakespeare's complete works
- Experiment with model configurations (layers, heads, embedding sizes)
- Generate character-level Shakespeare-style text
Features: 
- Trained using nanoGPT architecture
- Sampled text using temperature and max token strategies
- Generated text saved in generated_nanogpt.txt

# Part 3: Fine-Tuning DistilGPT2 using HuggingFace
Goals: 
- Fine-tune pre-trained DistilGPT2 on custom text (WikiText)
- Create controlled text generation with prompt-based inference
- Experiment with decoding controls
Libraries:
- transformers
- datasets
- accelerate
Features:
- Implemented supervised fine-tuning
- Controlled generation using temperature, top-p, max_tokens
- Saved model checkpoint and generated samples

# How to Run
Install Dependencies: 
pip install torch torchvision torchaudio tqdm transformers datasets accelerate

Train LeNet:
python train_cifar100.py
python eval_cifar100.py --load ./outputs/model_best.pth.tar

Train nanoGPT: 
python train.py --device=cpu
python sample.py --device=cpu

Fine-Tune DistilGPT2: 
python distilgpt2_sft_cpu.py --data data.csv --mode train
python distilgpt2_sft_cpu.py --mode gen

# Key Skills 
- Deep Learning (CNNs, transformers, GPT models)
- PyTorch model building and training
- HuggingFace fine-tuning and controlled text generation
- Understanding of model size, speed, and efficiency
- Experimentation with hyperparameters and inference tuning

# Author: Kavya Akkina



