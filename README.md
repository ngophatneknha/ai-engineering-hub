# AI Engineering, Autonomous Agents and LLM Master Repository

Kho luu tru tri thuc, ma nguon va he thong thuc hanh toan dien ve AI Engineering, Large Language Models (LLM), He thong Tac tu Tu hanh (Autonomous Agents), He thong Truy xuat Tang cuong (RAG) va Quy trinh Van hanh Mo hinh Hoc may (MLOps).

Toan bo ma nguon, giao trinh va tai lieu duoc dong goi doc lap, dong bo truc tiep tren may tinh ca nhan va he thong quan ly phien ban GitHub phuc vu muc tieu nghien cuu sau, thuc hanh thuc te va xay dung san pham production.

---

## 1. Tong Quan va Kien Truc He Thong

Kho luu tru duoc to chuc thanh 2 phan chinh:
1. Thu muc goc (Root): 93+ du an thuc chien doc lap ve OCR, Chat UI, Reasoning UI, RAG don/da phuong thuc, danh gia mo hinh va he thong Multi-Agent MCP.
2. Thu muc `curated-courses/`: 50 bo giao trinh, framework cot loi va kho du an mo nguon tieu bieu nhat the gioi duoc phan loai theo 8 tru cot kien thuc.

```
AI-ENGINEERING-HUB/
|-- [93+ Root Projects]             # Cac ung dung, vi du thuc hanh doc lap
|-- curated-courses/                # 50 kho giao trinh, thu vien va du an tich hop
|   |-- aie-book-chiphuyen/         # AI Engineering Book cua Chip Huyen
|   |-- langchain-framework/       # Framework LangChain cot loi
|   |-- langgraph-orchestration/    # Framework LangGraph dieu phoi agent
|   |-- crewai-multi-agent/         # Framework CrewAI cho he da agent
|   |-- qdrant-vector-database/     # Vector database Qdrant viet bang Rust
|   |-- ragas-eval-framework/       # Framework danh gia dinh luong RAG
|   |-- ollama-local-llms/          # Cong cu chay local LLM
|   |-- awesome-mcp-servers-catalog/# Danh muc Model Context Protocol servers
|   |-- mcp-reference-servers/      # Cac server mau Model Context Protocol
|   |-- superpowers-coding-skills/  # Framework skill va workflow cho coding agent
|   |-- claude-code-agents-suite/   # Bo agent dieu phoi cho Claude Code
|   |-- claude-mem-plugin/          # Plugin bo nho phien lam viec
|   |-- claude-flow-orchestration/  # Framework ruflo dieu phoi multi-agent
|   |-- planning-with-files/        # Quan ly tien do va context bang Markdown
|   |-- claude-scientific-skills/   # Ky nang nghien cuu khoa hoc va phan tich
|   |-- claude-code-guide/          # Huong dan thiet lap Claude Code va SKILL.md
|   |-- oh-my-claudecode/           # Bo agent dung san cho workflow phuc tap
|   |-- ecc-agent-harness/          # Toi uu harness agent, memory va security
|   |-- teamai-cli-tencent/         # CLI chia se skill va MCP config trong team
|   |-- pi-desktop-agent-env/       # Moi truong desktop danh rieng cho AI agent
|   |-- openai-plugins/             # Kho plugins chinh thuc tu OpenAI
|   |-- diagram-design/             # Mau bieu do ky thuat chuan cho AI tools
|   |-- system-design-notes/        # Ghi chu kien truc he thong lon
|   |-- ultimate-ai-engineer-roadmap-2026/ # Lo trinh AI Engineer 2026
|   |-- microsoft-ml-for-beginners/ # 12 tuan hoc may co dien scikit-learn
|   |-- microsoft-ai-for-beginners/ # Giao trinh AI, Neural Nets, PyTorch/TF
|   |-- microsoft-generative-ai-for-beginners/ # 21 bai hoc xay dung app GenAI
|   |-- microsoft-ai-agents-for-beginners/ # 18 bai hoc chuyen sau ve AI Agents
|   |-- microsoft-autogen/          # Framework AutoGen cho he da agent
|   |-- karpathy-nn-zero-to-hero/   # Xay dung mang neural tu micrograd toi GPT
|   |-- karpathy-nanochat/          # Toan trinh huan luyen va giao dien chatGPT
|   |-- rasbt-llms-from-scratch/    # Tu lap trinh GPT tu dau bang PyTorch
|   |-- llm-course/                 # Lo trinh LLM Scientist va Engineer
|   |-- huggingface-llm-course/     # Giao trinh chinh thuc tu Hugging Face
|   |-- ai-engineering-from-scratch/# Transformer, Vector DB, RAG tu so 0
|   |-- rag-techniques-nirdiamant/  # Tong hop toan bo ky thuat RAG nang cao
|   |-- genai-agents/               # Cac mau kien truc agent thuc chien
|   |-- llama-index-core/           # Data framework ket noi LLM va du lieu
|   |-- awesome-llm-apps/           # 100+ ung dung RAG va Agent production
|   |-- claude-cookbooks/           # Huong dan tool use, structured output
|   |-- anthropic-agent-skills/     # Chuan dong goi ky nang cho Agent
|   |-- agentic-loop-engineering/   # Kien truc vong lap dieu khien phan xa
|   |-- deep-learning-paper-implementations/ # 60+ mo hinh PyTorch theo paper
|   |-- dive-into-deep-learning/    # Sach tuong tac D2L da framework
|   |-- deep-learning-book/         # Sach Deep Learning MIT Press
|   |-- prompt-engineering-guide/   # Cam nang ky thuat prompt chuan quoc te
|   |-- made-with-ml/               # Quy trinh MLOps, CI/CD dua mo hinh len prod
|   |-- learn-ai-engineering/       # Lo trinh he thong va ky nang AI Engineer
|   |-- ai-expert-roadmap/          # So do ky nang Machine Learning va AI
|   `-- awesome-machine-learning/   # Danh ba cong cu ML theo moi ngon ngu
|-- README.md                       # Tai lieu huong dan trung tam
`-- LICENSE                         # Giay phep su dung MIT
```

---

## 2. Lo Trinh Hoc Tap Theo 8 Giai Doan

### Giai Doan 1: Nen Tang Toan Hoc, Machine Learning va Deep Learning
Muc tieu: Nam vung dai so tuyen tinh, giai tich ma tran, xac suat thong ke, thuat toan hoc may co dien va mang neuron sau.
* `microsoft-ml-for-beginners`: Thuc hanh 12 tuan ve hoi quy, phan loai, clustering bang Scikit-Learn voi du lieu thuc te.
* `microsoft-ai-for-beginners`: Kien truc mang no-ron, Computer Vision voi CNN, xu ly ngon ngu voi RNN va Transformer bang PyTorch va TensorFlow.
* `karpathy-nn-zero-to-hero`: Tu viet engine vi phan tu dong (micrograd), mo hinh ngon ngu cap ky tu (makemore), den transformer decoder (nanoGPT).
* `deep-learning-book`: Nen tang ly thuyet kinh dien tu giao trinh MIT Press cua Ian Goodfellow, Yoshua Bengio va Aaron Courville.
* `dive-into-deep-learning`: Giao trinh tuong tac day du ma nguon chay tren PyTorch, JAX va TensorFlow.
* `deep-learning-paper-implementations`: Danh muc 60+ kien truc paper kinh dien duoc viet bang PyTorch thuan tu Phil Wang (lucidrains) nhu Diffusion, ViT, RoPE, FlashAttention.
* `awesome-machine-learning`: Tra cuu toan bo he sinh thai thu vien va cong cu hoc may tren the gioi.

### Giai Doan 2: Hieu Sau LLM va Tu Xay Dung Tu Dau (Build From Scratch)
Muc tieu: Hieu ro tung phep tinh ben trong Transformer, co che Attention, tokenization va quy trinh huan luyen LLM.
* `rasbt-llms-from-scratch`: Huong dan tung buoc lap trinh kien truc GPT tu so 0 bang PyTorch, chuan bi du lieu, pre-training, fine-tuning va danh gia.
* `karpathy-nanochat`: Toan bo pipeline huan luyen mo hinh ngon ngu nho gon den giao dien web chat truc tiep.
* `ai-engineering-from-scratch`: Tu code Vector Database, Attention Mechanism, Transformer Block, RAG va Self-Correction Agent khong phu thuoc thu vien cap cao.
* `llm-course`: Lo trinh toan dien ve LLM Scientist va Engineer tu Maxime Labonne gom Tokenizer, PEFT/LoRA, QLoRA, DPO, KTO, Quantization (GGUF, AWQ, GPTQ).
* `huggingface-llm-course`: Giao trinh chuan cua Hugging Face ve he sinh thai Transformers, Datasets, TRL va Accelerate.

### Giai Doan 3: Prompt Engineering va Phat Trien Ung Dung Generative AI
Muc tieu: Lam chu ky thuat thiet ke cau lenh nang cao va lap trinh ung dung tich hop LLM API.
* `prompt-engineering-guide`: Cam nang day du nhat ve Zero-shot, Few-shot, Chain-of-Thought, Directional Stimulus Prompting, ReAct va Re-reading.
* `microsoft-generative-ai-for-beginners`: 21 bai hoc thuc te xay dung ung dung text generation, chat application, search va image generation.
* `claude-cookbooks`: Huong dan thuc hanh tool use, structured output, vision, citation va evaluation tu Anthropic.
* `awesome-llm-apps`: Bo suu tap hon 100 ung dung production ve agent, bot tro ly va RAG co the chay ngay.
* `openai-plugins`: Kho luu tru tai lieu va mau tich hop plugin cho mo hinh ngon ngu.

### Giai Doan 4: He Thong Truy Xuat Tang Cuong va Vector Database (RAG & Evals)
Muc tieu: Xay dung he thong hoi dap tren tai lieu doanh nghiep, ket noi du lieu da nguon va danh gia dinh luong do chinh xac.
* `rag-techniques-nirdiamant`: Huong dan toan dien moi ky thuat RAG: Naive RAG, Advanced Chunking, Re-ranking, Contextual Compression, Parent Document Retrieval, Multi-Query, HyDE, Corrective RAG, Self-RAG va Graph RAG.
* `llama-index-core`: Framework hang dau de ingest, parse, index va ket noi du lieu doanh nghiep voi LLM.
* `qdrant-vector-database`: Co so du lieu vector hieu nang cao viet bang Rust, ho tro hybrid search va payload filtering.
* `ragas-eval-framework`: Framework tieu chuan danh gia dinh luong chat luong he thong RAG qua cac chi so Faithfulness, Answer Relevancy, Context Precision va Context Recall.
* 93+ Du an tai root: Cac du an mau ve Document Chat RAG, Video RAG, Fastest RAG Stack, ModernBERT RAG va Multimodal RAG.

### Giai Doan 5: Tac Tu Tu Hanh va Dieu Phoi He Da Agent (Autonomous Agents)
Muc tieu: Thiet ke he thong agent co kha nang lap ke hoach, su dung cong cu, ghi nho va phoi hop nhieu agent giai quyet tac vu phuc tap.
* `langchain-framework`: Framework pho bien nhat xay dung chuoi xu ly LLM, bo nho va ket noi cong cu.
* `langgraph-orchestration`: Framework dieu phoi agent co trang thai (stateful) duoi dang do thi chu trinh (cyclic graph), phu hop cac workflow lon can human-in-the-loop va time-travel debugging.
* `crewai-multi-agent`: Framework to chuc doi ngu agent theo vai tro (Role-Playing Autonomous Agents) voi co che delegation va task execution.
* `microsoft-ai-agents-for-beginners`: Khoa hoc 18 bai chuyen sau ve Agentic Design Patterns, Tool Use, Multi-Agent Communication, Protocols va Production Deployment.
* `microsoft-autogen`: Framework giao tiep da tac tu hang dau ho tro dynamic group chat va code execution.
* `genai-agents`: Mau thiet ke kien truc agent thuc chien: Planning, Long-term Memory, Reflection va Hierarchical Coordination.
* `agentic-loop-engineering`: Kien truc thiet ke cac vong lap agentic loop va kiem soat trang thai hoi dap.

### Giai Doan 6: Moi Truong Tac Tu, Ky Nang va Chuan Giao Thuc (Agent Tools & MCP)
Muc tieu: Tich hop giao thuc Model Context Protocol (MCP), dong goi ky nang agent va toi uu moi truong lam viec cho AI coding agents.
* `mcp-reference-servers`: Bo may chu tham chieu chinh thuc cho Model Context Protocol ket noi Filesystem, SQLite, Postgres, Git, Puppeteer va Fetch.
* `awesome-mcp-servers-catalog`: Danh ba tong hop tat ca may chu MCP production hien co.
* `anthropic-agent-skills`: Quy chuan ky thuat dong goi huong dan va kịch ban duoi dang agent skills tai su dung duoc.
* `superpowers-coding-skills`: Bo khung skill va quy trinh kiem thu phan mem nang cao cho coding agent.
* `claude-code-agents-suite`: He thong subagents tu dong hoa va phan cong nhiem vu cho Claude Code.
* `claude-mem-plugin`: Plugin ghi nho thong tin xuyen phien lam viec va nen ngu canh tu dong.
* `claude-flow-orchestration`: Framework ruFlo dieu phoi luong cong viec phuc tap cua nhieu agent.
* `planning-with-files`: Phuong phap lap ke hoach va ghi nho tien do du an thong qua cac tep Markdown co cau truc.
* `claude-scientific-skills`: Bo ky nang chuyen sau cho nghien cuu, phan tich so lieu va xu ly khoa hoc.
* `claude-code-guide`: Huong dan thiet lap moi truong, file cau hinh SKILL.md, agent definitions va lenh tuy bien.
* `oh-my-claudecode`: Bo cong cu tien ich mo rong cho Claude Code giup trien khai nhanh multi-agent workflows.
* `ecc-agent-harness`: Toi uu hoa harness cho agent ve mat bao mat, bo nho va quy trinh nghien cuu.
* `teamai-cli-tencent`: Cong cu dong bo va chia se ky nang, quy tac va cau hinh MCP trong nhom ky thuat.
* `pi-desktop-agent-env`: Moi truong desktop cuc bo xay dung cho agent lap trinh tu dong.
* `ollama-local-llms`: Chay mo hinh cuc bo phuc vu suy luan va kiem thu agent offline khong ton chi phi API.

### Giai Doan 7: Thiet Ke He Thong va Truc Quan Hoa Kien Truc (System Design)
Muc tieu: Nam vung kien truc phan mem quy mo lon, thiet ke he thong AI chiu tai va ve so do truc quan.
* `system-design-notes`: Tai lieu chi tiet ve kien truc he thong phan tan, database sharding, caching, message queues, rate limiting va cac cau hoi phong van he thong tieu bieu.
* `diagram-design`: Bo huong dan va mau thiet ke so do he thong chuyen nghiep danh cho AI coding tools va tai lieu ky thuat.

### Giai Doan 8: Lo Trinh Ky Nang, MLOps va Sach AI Engineering Chuyen Sau
Muc tieu: Chuyen hoa mo hinh tu moi truong thu nghiem thanh he thong san pham co quy trinh CI/CD, giam sat va van hanh tin cay.
* `aie-book-chiphuyen`: Tai lieu, bai tap va huong dan tu cuon sach AI Engineering noi tieng cua Chip Huyen, bao quat toan bo chu trinh san pham AI.
* `made-with-ml`: Giao trinh MLOps thuc chien tu Goku Mohandas: kiem thu du lieu, tracking voi MLflow, dong goi Docker, CI/CD GitHub Actions va serving.
* `ultimate-ai-engineer-roadmap-2026`: Lo trinh ky nang cap nhat moi nhat cho nam 2026.
* `learn-ai-engineering`: Ban do tong hop cac chu de kien thuc va kien truc can thiet cho ky su AI.
* `ai-expert-roadmap`: So do chi tiet huong di nghe nghiep trong linh vuc AI va Data Science.

---

## 3. Danh Sach 50 Khoa Hoc va Du An Trong `curated-courses/`

| STT | Ten Du An / Thu Muc | Don Vi / Tac Gia | Mo Ta Chuc Nang |
|---|---|---|---|
| 1 | `aie-book-chiphuyen` | Chip Huyen | Sach AI Engineering: Tai nguyen, ghi chu va phan tich he thong AI thuc te |
| 2 | `langchain-framework` | LangChain | Framework xay dung ung dung LLM, ket noi chuoi xu ly va bo nho |
| 3 | `langgraph-orchestration` | LangChain | Dieu phoi luong agent co trang thai duoi dang do thi chu trinh |
| 4 | `crewai-multi-agent` | CrewAI Inc | Framework dieu phoi nhom tac tu dong vai tro thuc hien cong viec production |
| 5 | `qdrant-vector-database` | Qdrant | Vector database toc do cao viet bang Rust ho tro hybrid search |
| 6 | `ragas-eval-framework` | Exploding Gradients | Bo cong cu danh gia dinh luong he thong RAG |
| 7 | `ollama-local-llms` | Ollama | Cong cu chay mo hinh ngon ngu lon cuc bo tren may tinh ca nhan |
| 8 | `awesome-mcp-servers-catalog` | punkpeye | Danh ba tong hop cac may chu Model Context Protocol |
| 9 | `mcp-reference-servers` | Model Context Protocol | Cac server mau chinh thuc ket noi Filesystem, Postgres, Git, SQLite |
| 10 | `superpowers-coding-skills` | obra | Framework bo sung ky nang va quy trinh lap trinh cho coding agent |
| 11 | `claude-code-agents-suite` | wshobson | Bo agent tu dong hoa dieu phoi nhieu tac vu cho Claude Code |
| 12 | `claude-mem-plugin` | thedotmack | Plugin ghi nho ngu canh xuyen suot cac phien lam viec |
| 13 | `claude-flow-orchestration` | ruvnet | Framework ruFlo dieu phoi quy trinh da agent phuc tap |
| 14 | `planning-with-files` | OthmanAdi | Quan ly ke hoach va luu tru trang thai lam viec bang Markdown |
| 15 | `claude-scientific-skills` | K-Dense-AI | Bo ky nang phuc vu nghien cuu khoa hoc va phan tich so lieu |
| 16 | `claude-code-guide` | zebbern | Huong dan cau hinh chi tiet Claude Code, SKILL.md va workflow |
| 17 | `oh-my-claudecode` | Yeachan-Heo | Bo tien ich mo rong va agent dung san |
| 18 | `ecc-agent-harness` | affaan-m | Bo khung toi uu hoa harness cho agent, quan ly bo nho va bao mat |
| 19 | `teamai-cli-tencent` | Tencent | Cong cu dong bo ky nang va cau hinh agent cho doi ngu ky thuat |
| 20 | `pi-desktop-agent-env` | vastsa | Moi truong desktop cuc bo duoc toi uu cho AI coding agents |
| 21 | `openai-plugins` | OpenAI | Kho luu tru tai lieu va plugins mau tu OpenAI |
| 22 | `diagram-design` | Cathryn Lavery | Thu vien thiet ke bieu do he thong danh cho AI tools |
| 23 | `system-design-notes` | liquidslr | Tong hop kien thuc thiet ke he thong phan tan quy mo lon |
| 24 | `ultimate-ai-engineer-roadmap-2026` | Prince Singh | Ban do lo trinh toan dien cho ky su AI cap nhat 2026 |
| 25 | `microsoft-ml-for-beginners` | Microsoft | 12 tuan hoc may co dien voi du lieu thuc qua Scikit-Learn |
| 26 | `microsoft-ai-for-beginners` | Microsoft | Kien truc mang no-ron, CV, NLP bang PyTorch va TensorFlow |
| 27 | `microsoft-generative-ai-for-beginners` | Microsoft | 21 bai hoc xay dung ung dung Generative AI |
| 28 | `microsoft-ai-agents-for-beginners` | Microsoft | 18 bai hoc kien truc he thong tac tu AI chuyen sau |
| 29 | `microsoft-autogen` | Microsoft | Framework giao tiep va hop tac da tac tu tu dong |
| 30 | `karpathy-nn-zero-to-hero` | Andrej Karpathy | Tu viet micrograd, makemore den kien truc GPT tu con so 0 |
| 31 | `karpathy-nanochat` | Andrej Karpathy | Quy trinh tron ven huan luyen va giao dien chatGPT thu nho |
| 32 | `rasbt-llms-from-scratch` | Sebastian Raschka | Ma nguon tu code GPT tu dau bang PyTorch va fine-tuning |
| 33 | `llm-course` | Maxime Labonne | Lo trinh LLM Scientist va Engineer: Tokenizer, LoRA, DPO |
| 34 | `huggingface-llm-course` | Hugging Face | Giao trinh chuan cua Hugging Face ve Transformers va PEFT |
| 35 | `ai-engineering-from-scratch` | Rohit Ghumare | Tu code Transformer, Attention, Vector DB, RAG khong dung framework |
| 36 | `rag-techniques-nirdiamant` | Nir Diamant | Cam nang toan bo ky thuat RAG tu co ban den Graph RAG |
| 37 | `genai-agents` | Nir Diamant | Cac mau kien truc tac tu thuc te: Memory, Planning, Multi-Agent |
| 38 | `llama-index-core` | Jerry Liu | Framework ket noi mo hinh voi du lieu doanh nghiep tuy bien |
| 39 | `awesome-llm-apps` | Shubham Saboo | Hon 100 ung dung RAG va Agent mau san sang trien khai |
| 40 | `claude-cookbooks` | Anthropic | Huong dan thuc hanh Tool use, RAG va evaluation voi Claude |
| 41 | `anthropic-agent-skills` | Anthropic | Chuan dong goi ky nang va kich ban cho agent |
| 42 | `agentic-loop-engineering` | Cobus Greyling | Thiet ke kien truc vong lap phan xa va dieu phoi trang thai |
| 43 | `deep-learning-paper-implementations` | Phil Wang (lucidrains) | 60+ kien truc paper duoc lap trinh bang PyTorch thuan |
| 44 | `dive-into-deep-learning` | D2L.ai | Sach tuong tac Deep Learning da framework kem code thuc hanh |
| 45 | `deep-learning-book` | Ian Goodfellow | Tom tat va muc luc doc truc tuyen giao trinh MIT Press |
| 46 | `prompt-engineering-guide` | DAIR.AI | Cam nang ky thuat prompt toan dien va tieu chuan |
| 47 | `made-with-ml` | Goku Mohandas | Quy trinh MLOps dua he thong hoc may len production |
| 48 | `learn-ai-engineering` | Ashish Pratap Singh | Lo trinh ky nang va kien truc he thong AI thuc chien |
| 49 | `ai-expert-roadmap` | AMAI-GmbH | So do ky nang tong quat cho linh vuc AI va Data Science |
| 50 | `awesome-machine-learning` | Joseph Misiti | Danh ba thu vien hoc may mo nguon theo tung ngon ngu lap trinh |

---

## 4. Danh Muc 93+ Du An Thuc Hanh Doc Lap Tai Root

Tai thu muc goc co 93+ du an doc lap, moi du an deu co ma nguon Python hoac TypeScript va tep README rieng:

### Nhom Co Ban (Beginner)
* `LaTeX-OCR-with-Llama`: Chuyen doi cong thuc toan tu anh sang ma LaTeX dung Llama 3.2 Vision.
* `llama-ocr`: Ung dung OCR chay offline 100% tren may tinh voi Streamlit.
* `gemma3-ocr`: Trich xuat van ban co cau truc dung mo hinh Gemma-3.
* `qwen-2.5VL-ocr`: OCR da ngon ngu dung Qwen 2.5 Vision-Language.
* `local-chatgpt with DeepSeek`: Giao dien mini-ChatGPT chay cuc bo dung DeepSeek-R1 va Chainlit.
* `local-chatgpt`: Giao dien tro chuyen cuc bo voi Llama 3.2.
* `deepseek-thinking-ui`: Giao dien truc quan hoa luong suy luan (Thinking Process) cua DeepSeek-R1.
* `qwen3-thinking-ui`: Giao dien suy luan dung Qwen3:4B va Streamlit.
* `streaming-ai-chatbot`: He thong truyen phat stream cau tra loi thoi gian thuc voi Motia.
* `simple-rag-workflow`: Quy trinh RAG co ban dung LlamaIndex va Ollama.
* `document-chat-rag`: Hoi dap tai lieu PDF dung Llama 3.3.
* `fastest-rag-stack`: Stack RAG toc do cao ket hop SambaNova, LlamaIndex va Qdrant.
* `github-rag`: Chat truc tiep voi ma nguon GitHub repositories chay cuc bo.
* `modernbert-rag`: Ung dung RAG su dung bo nhung ModernBERT hien dai.
* `llama-4-rag`: Kien truc RAG toi uu cho Meta Llama 4.
* `imagegen-janus-pro`: Sinh anh cuc bo dung mo hinh DeepSeek Janus-Pro 7B.
* `video-rag-gemini`: Chat va hoi dap noi dung video dung Gemini AI.
* `Website-to-API-with-FireCrawl`: Chuyen doi website bat ky thanh API du lieu co cau truc.
* `ai_news_generator`: He thong tu dong tong hop tin tuc AI dung CrewAI va Cohere.

### Nhom Trung Cap (Intermediate - RAG & MCP Nang Cao)
* `agentic_rag` & `agentic_rag_deepseek`: He thong RAG co kha nang tu kiem tra do phu hop cua tai lieu va tu sua truy van.
* `fast-agentic-rag-llamaindex`: Agentic RAG hieu nang cao voi LlamaIndex.
* `lightrag-multimodal`: Kien truc LightRAG ho tro du lieu van ban va hinh anh ket hop.
* `graphrag-orchestration`: Ket hop do thi tri thuc (Knowledge Graph) voi vector retrieval giup tra loi cau hoi mang tinh tong the.
* `cursor_linkup_mcp`: MCP server ket noi tim kiem web sau danh cho Cursor IDE.
* `eyelevel-mcp-rag`: May chu MCP cho he thong RAG tren tai lieu phuc tap.
* `llamaindex-mcp`: Client MCP cuc bo tich hop LlamaIndex.
* `mcp-agentic-rag`: RAG agentic duoc dong goi theo tieu chuan Model Context Protocol.
* `mcp-video-rag`: Video RAG su dung Ragie qua giao thuc MCP.
* `mcp-voice-agent`: Tac tu giong noi dung Firecrawl, Supabase va MCP.
* `sdv-mcp`: Dieu phoi sinh du lieu tong hop (Synthetic Data Vault) qua MCP.
* `stagehand x mcp-use`: Tu dong hoa trinh duyet web ket hop giua Stagehand va MCP.
* `eval-and-observability`: Giam sat va danh gia toan trinh RAG qua CometML Opik.
* `code-model-comparison`: So sanh kha nang sinh ma nguon giua cac mo hinh lon.

### Nhom Nang Cao (Advanced - Fine-Tuning & Multi-Agent)
* `DeepSeek-finetuning`: Huong dan tinh chinh DeepSeek dung Unsloth va Ollama tiet kiem VRAM.
* `Build-reasoning-model`: Xay dung mo hinh suy luan giong DeepSeek-R1 tu dau.
* `attention-is-all-you-need-impl`: Lap trinh chi tiet kien truc Transformer goc tu paper goc cua Google.
* `Multi-Agent-deep-researcher-mcp-windows-linux`: He thong nghien cuu sau su dung doi ngu da tac tu MCP ho tro ca Windows va Linux.
* `web-browsing-agent`: Tac tu duyet web tu dong voi CrewAI va Stagehand.
* `paralegal-agent-crew`: Doi ngu tro ly phap ly thong minh tich hop RAG.
* `notebook-lm-clone`: Phien ban clone day du cua Google NotebookLM voi RAG, trich dan va sinh podcast hoi thoai.
* `groundX-doc-pipeline`: Pipeline xu ly tai lieu doanh nghiep chuan quoc te.
* `ultimate-ai-assitant-using-mcp`: Giao dien Streamlit ket noi da may chu MCP Server.

---

## 5. Huong Dan Cai Dat va Su Dung

### Yeu Cau He Thong
* He dieu hanh: Windows 10/11, Ubuntu 20.04+, hoac macOS.
* Python: Phien ban 3.10 tro len (khuyen nghi Python 3.11).
* Git: Da cai dat va ho tro Git LFS.
* Bo nho RAM: Toi thieu 16 GB (khuyen nghi 32 GB cho cac mo hinh lon).
* GPU (tuy chon): NVIDIA GPU ho tro CUDA 12.x de chay mo hinh cuc bo.

### Thiet Lap Moi Truong
1. Tao moi truong ao dung Conda hoac venv:
```bash
conda create -n ai-master python=3.11 -y
conda activate ai-master
```

2. Cai dat PyTorch ho tro tang toc CUDA (vi du CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Cai dat cac thu vien nen tang hang dau:
```bash
pip install transformers datasets peft accelerate trl
pip install langchain langchain-community langgraph crewai llama-index
pip install qdrant-client ragas scikit-learn numpy pandas
```

4. Su dung va chay du an:
Moi thu muc con deu hoat dong doc lap. Di chuyen vao thu muc can chay va khoi dong:
```bash
cd curated-courses/rasbt-llms-from-scratch
pip install -r requirements.txt
python -m unittest
```

Hoac chay mot du an RAG bat ky o thu muc goc:
```bash
cd simple-rag-workflow
python app.py
```

---

## 6. Quy Dinh Dong Gop va Giay Phep

* Du an duoc to chuc duoi giay phep MIT License.
* Ma nguon tu cac kho mo duoc tich hop tuan thu giay phep goc cua tung tac gia.
* Moi su dong gop hoac phat trien them deu nham muc dich nghien cuu khoa hoc, chia se cong dong va nang cao nang luc AI Engineering.
