# Benchmark and Scaling

There are many factors that affect the training performance. We provide some benchmark results here.

* Training llama-3b on 4x RTX 3090: ~6300 tokens per second. [Training Report](https://wandb.ai/autoai-org/fmengine/reports/OpenLlama-3b-Chat--Vmlldzo1NTAxMTYy) [Configuration and Monitoring](https://wandb.ai/autoai-org/fmengine/runs/3ddwtzyl?workspace=user-xzyaoi).

## Scaling to Clusters of GPUs

We conduct scaling experiments on up to 160 NVIDIA A100 40G GPUs. We thank [Juelich Supercomputing Center](https://www.fz-juelich.de/de) and [Ontocord](https://www.ontocord.ai/) for their generous support in providing the computing resources.

### Benchmark 1: Fine-tuning Llama-7B

| Train Batch Size | Micro Batch Size | Sequence Length | 
| --- | --- | --- |
| 512 | 8 | 2048 |

* Total Tokens/second

<iframe width="600" height="371" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSkxCvnWKnlAxQWBI2F34ODYxTCtpNT-d1cuY-_s4myZ79wtKh8kV1rLqiFeYjvJXURwdLQFk2ps73Z/pubchart?oid=758643514&amp;format=interactive"></iframe>

* Tokens/second/GPU

<iframe width="600" height="371" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSkxCvnWKnlAxQWBI2F34ODYxTCtpNT-d1cuY-_s4myZ79wtKh8kV1rLqiFeYjvJXURwdLQFk2ps73Z/pubchart?oid=1702179548&amp;format=interactive"></iframe>

### Benchmark 2: Fine-tuning Llama-70B

| Train Batch Size | Micro Batch Size | Sequence Length | 
| --- | --- | --- |
| 1024 | 4 | 4096 |

* Total Tokens/second

<iframe width="600" height="371" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSkxCvnWKnlAxQWBI2F34ODYxTCtpNT-d1cuY-_s4myZ79wtKh8kV1rLqiFeYjvJXURwdLQFk2ps73Z/pubchart?oid=612904704&amp;format=interactive"></iframe>

* Tokens/second/GPU

<iframe width="600" height="371" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSkxCvnWKnlAxQWBI2F34ODYxTCtpNT-d1cuY-_s4myZ79wtKh8kV1rLqiFeYjvJXURwdLQFk2ps73Z/pubchart?oid=1963687240&amp;format=interactive"></iframe>