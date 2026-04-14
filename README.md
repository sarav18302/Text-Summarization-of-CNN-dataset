# Text Summarization of CNN dataset
## Task 4: Analysis & Creative Thinking

### Design Decisions:
- Built the parser as a recursive traversal using PyTorch’s `named_children()` so it works uniformly across models without hardcoding architecture-specific logic.  

- Each module is converted into a consistent dictionary structure, and recursion naturally captures the hierarchical organization of the network.  

- Avoided any model-specific assumptions, allowing the same parser to work for GPT-2, TinyLlama, Phi, and other architectures without modification.  

- Prioritized clarity and generality over performance, even though this results in creating many small objects during traversal.  

- Introduced a `max_depth` parameter to keep the output readable, trading off some low-level detail for better usability.  

- Ensured the output is fully JSON-serializable so it can be stored, compared across runs, or reused in downstream tasks like model merging and analysis.

### Working Conditions:
- **Hardware Setup:** Used Google Colab with a single GPU (T4/A100 depending on availability), ~12–16 GB GPU VRAM, and ~12 GB system RAM.  

- **Execution Time:**  
  - Visualized in notebook  

- **Key Bottleneck:** Loading multiple models simultaneously caused frequent Colab crashes due to GPU/CPU memory limits.  

- **Solution:** Implemented **lazy loading**, where models are loaded only when needed during evaluation instead of keeping all models in memory.  

- **Checkpoint Strategy:** Saved intermediate artifacts (LoRA adapter, merged model, averaged model) to disk, allowing models to be reloaded on demand instead of recomputed.  

- **Memory Optimization:** Used 4-bit quantization and explicit memory cleanup (`del`, `torch.cuda.empty_cache()`, `gc.collect()`) to manage limited resources effectively.

### Extensibility & Scalability:
- I structured the system using modular classes and abstract base classes (for model handling, parsing, and training), making it easy to extend to additional models or even entirely new architectures without changing core logic.  

- For ~10 models, this design scales well by using a config-driven setup (model name → path → type) along with lazy loading, ensuring only one model is in memory at a time.  

- I also incorporated checkpointing and a results ledger approach so intermediate outputs (metrics, merged models, adapters) can be reused without recomputation.  

- At ~50 models, evaluation time becomes the primary bottleneck, which can be addressed by parallelizing evaluation using multiprocessing or frameworks like Ray.  

- Storage becomes a concern at scale, so instead of saving full model checkpoints, I store LoRA adapters and reconstruct merged models on demand.
- For cross-architecture support (e.g., Llama vs Qwen), the abstract interfaces allow plugging in different model loaders and tokenizers without changing the core pipeline.  

### Creativity & Future Vision:
- I have designed the system as a fully modular pipeline using well-defined classes and Pydantic-based configs for each stage (parsing, evaluation, fine-tuning, merging), enabling easy extension, validation, and reproducibility across different models and experiments.  

- Instead of generic fine-tuning, I would use a failure-driven + active learning loop: identify low-performing samples, cluster them into failure modes, fine-tune small LoRA patches on only those subsets, and dynamically merge/unmerge them while tracking performance drift.  

- For model composition, I would move beyond uniform weight averaging and apply layer-wise adaptive merging, where earlier layers rely more on the base model and later layers incorporate more task-specific updates from LoRA or fine-tuned weights.  

- **Unique twist I would add:** Before any weight update, I would insert a prompt-space optimization step (few-shot / template search) to verify whether the performance gap is actually due to prompting rather than model capability, reducing unnecessary fine-tuning.


### Honest Reflection:
- The overall pipeline design and modularization using classes and abstract interfaces felt relatively straightforward, as it followed standard software design principles.  

- The more challenging parts were working with LoRA fine-tuning, model merging, and understanding internal model structures (e.g., `named_children()`), especially ensuring everything worked correctly under memory constraints.  

- I relied on external documentation for implementing LoRA, merging strategies, and understanding model internals, since these are not trivial to implement from scratch.  

- I also used LLM assistance for organizing code structure, improving readability, and exploring model composition techniques such as LERP and SLERP.  

- However, the final system design — particularly the modular architecture, evaluation pipeline, and ideas around failure-driven improvement and adaptive merging — reflects my own reasoning and integration of these concepts into a cohesive workflow.
