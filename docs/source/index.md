# FMEngine

FMEngine is a utility library for training very large foundation models. The goal of fmengine is to provide a 

* **Ergonomic** interface for training foundation models. It is sufficient easy for a beginner to use, but also provides enough flexibility for advanced users to customize their training.
* **Efficient** optimizations built in. FMEngine is equipped with [Flash Attention](https://github.com/Dao-AILab/flash-attention) and various fused ops to accelerate training.
* **HPC-friendly** installation with pre-built docker and singularity/apptainer containers. FMEngine is mainly designed and tested on [Slurm](https://slurm.schedmd.com/) clusters. We provide starter scripts for running FMEngine on Slurm clusters.
* **Compatible** with existing frameworks and tools, particularly with [HuggingFace](https://huggingface.co). Since FMEngine is built with [DeepSpeed](https://deepspeed.ai), it is also compatible with all DeepSpeed features.

For now, FMEngine supports two families of models: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) and [LLama](https://ai.meta.com/blog/large-language-model-llama-meta-ai/). 

| Model | #Params | #Layers | #Heads |  #Dim | Pretrained Checkpoint | Flash Attention |
| --- | --- | --- | --- | --- | --- | --- |
| Pythia-160M | 85M | 12 | 12 | 768 | [Download](https://drive.google.com/file/d/1QZNSCMEIldyUVe0ZqMRjlZJZ3WA8KAAE/view?usp=drive_link) | Yes |
| Pythia-1.4B | 1.2B | 24 | 16 | 2048 | [Download](https://drive.google.com/file/d/16EB64Y0YmYpcr022EO4gxmDszGkLHl8a/view?usp=drive_link) | Yes |
| Pythia-2.8B | 2.5B | 32 | 32 | 2560 | [Download](https://drive.google.com/file/d/1Q03nrVOP7rLDrADgQsWA_BM8_ojD2qbE/view?usp=drive_link) | Yes |
| OpenLlama-3B | tba | tba | tba | tba | [Download](https://drive.google.com/file/d/1EYTaPXoBrAk4OTXqNug2N62poCCsv0Ru/view?usp=drive_link) | Yes |
| Llama-2-70b | tba | tba | tba | tba | tba | Yes |


## Acknowledgement

FMEngine is primarily implemented at the [Efficient Architecture and Systems Labs](https://systems.ethz.ch/research/easl.html).

![https://systems.ethz.ch/research/easl.html](https://systems.ethz.ch/research/easl/_jcr_content/par/textimage_842607556/image.imageformat.textsingle.745562631.png)

```{toctree}
:hidden:

quickstart
distributed
add_new_model
references
```
