# FMEngine

<img src="https://github.com/eth-easl/fmengine/blob/init/assets/logo.png" width="240"/>

FMEngine is a utility library for training very large foundation models. The goal of fmengine is to provide the following:

* **Ergonomic** interface for training foundation models. It is sufficient easy for a beginner to use, but also provides enough flexibility for advanced users to customize their training.
* **Efficient** optimizations built in. FMEngine is equipped with [Flash Attention](https://github.com/Dao-AILab/flash-attention) and various fused ops to accelerate training.
* **HPC-friendly** installation with pre-built docker and singularity/apptainer containers. FMEngine is mainly designed and tested on [Slurm](https://slurm.schedmd.com/) clusters. We provide starter scripts for running FMEngine on Slurm clusters.
* **Compatible** with existing frameworks and tools, particularly with [HuggingFace](https://huggingface.co). Since FMEngine is built with [DeepSpeed](https://deepspeed.ai), it is also compatible with all DeepSpeed features.

## Acknowledgement

FMEngine is primarily implemented and maintained by the [Efficient Architecture and Systems Labs @ ETH Zurich](https://systems.ethz.ch/research/easl.html).

<a href="https://systems.ethz.ch/research/easl.html"><img src="https://systems.ethz.ch/research/easl/_jcr_content/par/textimage_842607556/image.imageformat.textsingle.745562631.png" width="120"></a>

We thank our friends for their generous support:

<a href="https://laion.ai/"><img src="https://avatars.githubusercontent.com/u/92627801?s=200&v=4" width="80"/></a>
<a href="https://github.com/ontocord/"><img src="https://avatars.githubusercontent.com/u/8900094?v=4" width="80"/></a>
<a href="https://huggingface.co/Multi-Domain-Expert-Learning"><img src="https://aeiljuispo.cloudimg.io/v7/https://cdn-uploads.huggingface.co/production/uploads/5fc6879e1c5ee87b1164876d/IoeynCnY_cMdjPAzrdU-2.jpeg?w=200&h=200&f=face" width="80"/></a>
<a href=""><img src="https://www.fz-juelich.de/static/media/Logo.2ceb35fc.svg" width="200"/></a>

### Contributors

* [@xzyaoi](https://github.com/xzyaoi/)
* [@LorrinWWW](https://github.com/LorrinWWW)
* [@fishiu](https://github.com/fishiu): Scaling up Low Rank Adapters + Longer Contexts
* [@Taishi-N324](https://github.com/Taishi-N324/): Benchmarking and Scaling Experiments
* [@chiennv2000](https://github.com/chiennv2000): Mistral Integration
