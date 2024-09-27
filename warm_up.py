import math

def warmup_decay_cosine(warmup_steps, loop_steps):
    '''
    warmup_steps不被包含在第一个loop中
    loop_steps: 每个cosine周期的step数
    '''
    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1e-4, warmup_steps))
        step = step - warmup_steps
        rate = (step // loop_steps + 1)
        # progress: cosine部分一个周期内的进度 0~1
        step = step % loop_steps
        progress = float(step) / float(max(1, loop_steps))
        lr = 0.5 * (1.0 + math.cos(math.pi * progress)) / rate
        return min(lr, 1)
    return fn