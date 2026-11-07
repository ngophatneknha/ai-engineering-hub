# Hugging Face LLM Course 🚀🤗

**Official Course:** [https://huggingface.co/learn/llm-course/](https://huggingface.co/learn/llm-course/)  
**Hugging Face GitHub:** [https://github.com/huggingface](https://github.com/huggingface)

A comprehensive, hands-on, community-driven course covering the modern Large Language Model (LLM) lifecycle from architecture to alignment, evaluation, and deployment.

---

## 📚 Curriculum Breakdown

### Unit 1: Introduction to LLMs & Tokenization
* Understanding autoregressive language modeling
* BPE (Byte Pair Encoding), WordPiece, and SentencePiece tokenizers
* Hands-on tokenizer training and token ID inspection using Hugging Face `tokenizers` and `transformers`

### Unit 2: Transformer Architecture & Attention Deep-Dive
* Self-Attention, Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA)
* Rotary Positional Embeddings (RoPE) and FlashAttention optimizations
* KV Cache mechanism for high-throughput text generation

### Unit 3: Parameter-Efficient Fine-Tuning (PEFT)
* Fine-tuning bottlenecks and why full fine-tuning is rarely practical
* LoRA (Low-Rank Adaptation) and QLoRA (4-bit Quantized LoRA)
* Prefix Tuning, Prompt Tuning, and $(IA)^3$
* Hands-on fine-tuning scripts using `peft` and `trl` (Transformer Reinforcement Learning)

### Unit 4: Preference Alignment & Post-Training
* Instruction Tuning (Supervised Fine-Tuning - SFT)
* RLHF (Reinforcement Learning from Human Feedback) with PPO
* DPO (Direct Preference Optimization) and KTO (Kahneman-Tversky Optimization)
* ORPO (Odds Ratio Preference Optimization)

### Unit 5: Quantization & Compression
* Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT)
* Techniques: bitsandbytes (int8, fp4, nf4), AWQ (Activation-aware Weight Quantization), GPTQ, GGUF
* Running local quantized models on consumer GPUs and CPUs (llama.cpp, Ollama, vLLM)

### Unit 6: Evaluation & Benchmarks
* Automated benchmarks: MMLU, GSM8K, HumanEval, MT-Bench
* LLM-as-a-Judge evaluations and automated grading pipelines
* Using Lighteval and LM-Evaluation-Harness

### Unit 7: Production Inference & Deployment
* High-performance serving with vLLM, TGI (Text Generation Inference), and TensorRT-LLM
* Speculative Decoding and continuous batching
* Building REST APIs and OpenAI-compatible endpoints

---

## 🛠️ Essential Hugging Face Libraries
* [`transformers`](https://github.com/huggingface/transformers): State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.
* [`datasets`](https://github.com/huggingface/datasets): The largest hub of ready-to-use NLP/ML datasets.
* [`peft`](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning methods.
* [`trl`](https://github.com/huggingface/trl): Train transformer language models with reinforcement learning and direct preference optimization.
* [`accelerate`](https://github.com/huggingface/accelerate): A simple way to train and use PyTorch models with multi-GPU, TPU, and mixed precision.
