# Quick Start

## Installation

We are not providing any pre-built releases at the moment, due to the complexity of the requirements and dependencies. A singularity image is provided, with all environment bundled in and can be used as a starting point.

Currently, fmengine supports two model family: GPT-NeoX and Llama.

## Training preparation

* *Prepare checkpoints*. As the first step, you will need to split a large model checkpoint into smaller pieces for each layer. This can be done by running the following command:

```bash
python scripts/conversions/llama/from_hf.py \
--model_name_or_path meta-llama/Llama-2-7b-hf  \
--output_dir path_to_outdir/llama2-7b \
--mp_world_size 1
```

You can download pre-configured checkpoints here: [Google Drive](https://drive.google.com/drive/folders/1rKfR-rJFsV5VFpC_Y9FjUynDUdkg45Lk?usp=sharing).

* *Prepare datasets*. We now only supports `.jsonl` format, which is a list of json objects, each of which contains a `text` field. For example, a sample of the dataset can be:

```json
{"text": "I love this movie!"}
{"text": "I hate this movie!"}
{"text": "I don't know."}
```

## Training

In `/scripts`, we show some examples of training scripts, for example, to finetune a pythia-2.8b model, you can run the following command:
``` bash
deepspeed --num_gpus 4 --num_nodes 1 starter.py \
    --output_dir .cache/models \
    --init_ckpt /pretrained_weights/pythia-160m-deduped \
    --data_path /datasets/quantitative_natural_instructions/train/all.train.jsonl \
    --max_seq_len 1024 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 100 \
    --log_steps 1 \
    --pipe_parallel_size 1 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ./configs/pythia.json
```

You are also advised to read `./configs/pythia.json` for the deepspeed configuration, which convers the learning rate, batch size, etc.

## Supported Models

(we only tried finetuning but not pretraining - but it should work)

| Model | #Params | #Layers | #Heads |  #Dim | Pretrained Checkpoint | Flash Attention |
| --- | --- | --- | --- | --- | --- | --- |
| Pythia-160M | 85M | 12 | 12 | 768 | [Download](https://drive.google.com/file/d/1QZNSCMEIldyUVe0ZqMRjlZJZ3WA8KAAE/view?usp=drive_link) | Yes |
| Pythia-1.4B | 1.2B | 24 | 16 | 2048 | [Download](https://drive.google.com/file/d/16EB64Y0YmYpcr022EO4gxmDszGkLHl8a/view?usp=drive_link) | Yes |
| Pythia-2.8B | 2.5B | 32 | 32 | 2560 | [Download](https://drive.google.com/file/d/1Q03nrVOP7rLDrADgQsWA_BM8_ojD2qbE/view?usp=drive_link) | Yes |
| OpenLlama-3B | tba | tba | tba | tba | [Download](https://drive.google.com/file/d/1EYTaPXoBrAk4OTXqNug2N62poCCsv0Ru/view?usp=drive_link) | Yes |

### Multi-host training

FMEngine support multi-host training with deepspeed. To run multi-host training, you need to install [`pdsh`](https://github.com/chaos/pdsh) first, by running the following command:

```bash
git clone https://github.com/chaos/pdsh.git
cd pdsh
./configure --enable-static-modules --without-rsh --with-ssh --without-ssh-connect-timeout-option --prefix=/your/preferred/path
make
make install
```

If you have root access, it might be easier.

## References

- [Deepspeed Configuration References](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options)
