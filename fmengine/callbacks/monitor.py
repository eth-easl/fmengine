from fmengine.utils import logger_rank0

def speed_monitor(elapsed_time, current_step, current_loss, configs):
    if current_step % configs['trainer']['log_steps'] == 0:
        step_time = elapsed_time/current_step
        logger_rank0.info(f"Step={current_step:>6}, loss={current_loss.item():.4f}, {step_time:.2f} s/it, {configs['deepspeed']['train_batch_size'] * configs['trainer']['max_seq_len']/step_time:.2f} tokens/s")