# 🔬 Deep Learning Paper Implementations (PyTorch)

> **Kho tổng hợp mã nguồn thực thi 60+ bài báo khoa học (Paper) Deep Learning kinh điển từ tác giả Phil Wang ([lucidrains](https://github.com/lucidrains)).**  
> Các kiến trúc được viết bằng PyTorch thuần túy, sạch sẽ, tối ưu hóa cao và dễ dàng tái sử dụng trong các dự án nghiên cứu và sản xuất.

---

## 🎨 1. Generative Models, Diffusion & Text-to-Image
* [**denoising-diffusion-pytorch**](https://github.com/lucidrains/denoising-diffusion-pytorch): Implementation của Denoising Diffusion Probabilistic Models (DDPM) 1D/2D trong PyTorch.
* [**DALLE-pytorch**](https://github.com/lucidrains/DALLE-pytorch): Tái hiện mô hình DALL-E của OpenAI (Zero-shot Text-to-Image generation) sử dụng dVAE và Discrete VAE.
* [**DALLE2-pytorch**](https://github.com/lucidrains/DALLE2-pytorch): Triển khai DALL-E 2 sử dụng Diffusion Prior và Unet Decoder.
* [**imagen-pytorch**](https://github.com/lucidrains/imagen-pytorch): Mô hình Text-to-Image Photorealistic Imagen của Google Research với T5 text encoder.
* [**muse-maskgit-pytorch**](https://github.com/lucidrains/muse-maskgit-pytorch): Text-to-Image Generation nhanh vượt trội sử dụng Masked Generative Video/Image Transformer.
* [**make-a-video-pytorch**](https://github.com/lucidrains/make-a-video-pytorch): Kiến trúc Text-to-Video generation của Meta AI.

---

## 👁️ 2. Vision Transformers & Multimodal
* [**vit-pytorch**](https://github.com/lucidrains/vit-pytorch): Triển khai Vision Transformer (ViT) từ bài báo *"An Image is Worth 16x16 Words"* cùng 15+ biến thể (Cross-ViT, MobileViT, Masked Autoencoder, Simple-ViT).
* [**perceiver-pytorch**](https://github.com/lucidrains/perceiver-pytorch): Kiến trúc Perceiver & Perceiver IO của DeepMind xử lý mọi loại dữ liệu (hình ảnh, âm thanh, point clouds, video) với độ phức tạp tuyến tính.
* [**flamingo-pytorch**](https://github.com/lucidrains/flamingo-pytorch): Mô hình Visual Language Model (VLM) ít mẫu (few-shot) của DeepMind.
* [**cross-attention-control**](https://github.com/lucidrains/cross-attention-control): Điều khiển Prompt-to-Prompt image editing trong Text-to-Image models.

---

## ⚡ 3. Transformer Architectures & Attention Innovations
* [**x-transformers**](https://github.com/lucidrains/x-transformers): Thư viện Transformer toàn diện nhất PyTorch tích hợp tất cả các cải tiến hiện đại: RoPE, ALiBi, FlashAttention, SwiGLU, RMSNorm, Multi-Query/Grouped-Query Attention.
* [**rotary-embedding-torch**](https://github.com/lucidrains/rotary-embedding-torch): Triển khai Rotary Position Embedding (RoPE) chuẩn xác được sử dụng trong Llama 2/3, Mistral, Qwen, DeepSeek.
* [**sinkhorn-transformer**](https://github.com/lucidrains/sinkhorn-transformer): Transformer với cơ chế phân cụm tự chú ý Sinkhorn giảm độ phức tạp xuống $O(N \log N)$.
* [**linear-attention-transformer**](https://github.com/lucidrains/linear-attention-transformer): Cơ chế Linear Attention tính toán trong thời gian tuyến tính $O(N)$.
* [**flash-attention-minimal**](https://github.com/lucidrains/flash-attention-minimal): Triển khai FlashAttention ngắn gọn dễ hiểu phục vụ nghiên cứu và học thuật.

---

## 🎵 4. Audio, Voice & Speech Generation
* [**audiolm-pytorch**](https://github.com/lucidrains/audiolm-pytorch): Ngôn ngữ sinh âm thanh chất lượng cao của Google AudioLM.
* [**musiclm-pytorch**](https://github.com/lucidrains/musiclm-pytorch): Hệ thống sinh âm nhạc từ văn bản (Text-to-Music) của Google.
* [**soundstream-pytorch**](https://github.com/lucidrains/soundstream-pytorch): Bộ mã hóa thần kinh âm thanh nén Neural Audio Codec (SoundStream & EnCodec).
* [**voicebox-pytorch**](https://github.com/lucidrains/voicebox-pytorch): Mô hình sinh giọng nói đa năng linh hoạt của Meta AI.

---

## 🧠 5. Large Language Models & Alignment
* [**PaLM-pytorch**](https://github.com/lucidrains/PaLM-pytorch): Triển khai Pathways Language Model (PaLM 540B) của Google với SwiGLU và RoPE.
* [**PaLM-rlhf-pytorch**](https://github.com/lucidrains/PaLM-rlhf-pytorch): Đào tạo mô hình PaLM kết hợp Reinforcement Learning with Human Feedback (RLHF).
* [**feedback-transformer-pytorch**](https://github.com/lucidrains/feedback-transformer-pytorch): Cơ chế hồi tiếp biểu diễn cấp cao cho autoregressive sequence modeling.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng nhanh

Bất kỳ mô hình nào của `lucidrains` đều có thể cài đặt dễ dàng qua `pip`:
```bash
pip install x-transformers vit-pytorch denoising-diffusion-pytorch
```

Ví dụ chạy nhanh Vision Transformer:
```python
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)
preds = v(img) # (1, 1000)
print("Predictions shape:", preds.shape)
```
