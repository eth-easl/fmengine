from fmengine.utils import logger_rank0
from fmengine.utils.monitor import rank0_log


def speed_monitor(elapsed_time, current_step, current_loss, configs, engine):
    if current_step % configs["trainer"]["log_steps"] == 0:
        logger_rank0.info(f"Step={current_step:>6}, Loss={current_loss.item():.4f}")
        logger_rank0.info(f"{elapsed_time:.2f} s/step")
        logger_rank0.info(
            f"{configs['deepspeed']['train_batch_size'] * configs['trainer']['max_seq_len']/elapsed_time:.2f} tokens/s"
        )


def wandb_monitor(elapsed_time, current_step, current_loss, configs, engine):
    tps = (
        configs["deepspeed"]["train_batch_size"]
        * configs["trainer"]["max_seq_len"]
        / elapsed_time
    )
    consumed_tokens = (
        current_step
        * configs["deepspeed"]["train_batch_size"]
        * configs["trainer"]["max_seq_len"]
        // 1_000
    )
    rank0_log(
        {
            "loss": current_loss.item(),
            "lr": engine.optimizer.param_groups[0]["lr"],
            "step": current_step,
            "tokens_per_second": tps,
            "step_time": elapsed_time,
            "consumed_tokens": consumed_tokens,
        }
    )
