from fmengine.utils import logger_rank0

def speed_monitor(elapsed_time, current_step, current_loss, configs):
    if current_step % configs['trainer']['log_steps'] == 0:
        logger_rank0.info(f"Step={current_step:>6}, loss={current_loss.item():.4f}, {elapsed_time/current_step:.2f} s/it")