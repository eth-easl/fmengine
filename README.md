# FMEngine

<img src="/assets/logo.png" width="240"/>

FMEngine is a utility library for training very large foundation models. The goal of fmengine is to provide the following:

* **Ergonomic** interface for training foundation models. It is sufficient easy for a beginner to use, but also provides enough flexibility for advanced users to customize their training.
* **Efficient** optimizations built in. FMEngine is equipped with [Flash Attention](https://github.com/Dao-AILab/flash-attention) and various fused ops to accelerate training.
* **HPC-friendly** installation with pre-built docker and singularity/apptainer containers. FMEngine is mainly designed and tested on [Slurm](https://slurm.schedmd.com/) clusters. We provide starter scripts for running FMEngine on Slurm clusters.
* **Compatible** with existing frameworks and tools, particularly with [HuggingFace](https://huggingface.co). Since FMEngine is built with [DeepSpeed](https://deepspeed.ai), it is also compatible with all DeepSpeed features.

## Acknowledgement

FMEngine is primarily implemented and maintained by the [Efficient Architecture and Systems Labs @ ETH Zurich](https://systems.ethz.ch/research/easl.html).

<a href="https://systems.ethz.ch/research/easl.html"><img src="https://systems.ethz.ch/research/easl/_jcr_content/par/textimage_842607556/image.imageformat.textsingle.745562631.png" width="120"></a>
