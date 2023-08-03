# FMEngine

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