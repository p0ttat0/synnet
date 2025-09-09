import time
import sys


class ProgressBar:
    def __init__(self, total_steps, epochs, bar_length=50):
        self.total_steps = total_steps
        self.epochs = epochs
        self.bar_length = bar_length
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self, epoch, current_step, metrics):
        if self.start_time is None:
            self.start()

        percent = (current_step + 1) / self.total_steps
        filled_length = int(self.bar_length * percent)
        bar = '█' * filled_length + ' ' * (self.bar_length - filled_length)

        time_elapsed = time.time() - self.start_time
        time_per_step = time_elapsed / (current_step + 1)
        time_remaining = (self.total_steps - (current_step + 1)) * time_per_step

        metrics_str = " | ".join([f"{name}: {value:.4f}" for name, value in metrics.items()])

        sys.stdout.write(
            f"\rEpoch {epoch+1}/{self.epochs} [{current_step}/{self.total_steps}] "
            f"[{bar}] {percent*100:.1f}% - "
            f"ETA: {time.strftime('%H:%M:%S', time.gmtime(time_remaining))} | "
            f"{metrics_str}"
        )
        sys.stdout.flush()

    def end(self):
        time_elapsed = time.time() - self.start_time
        print(f'\n\nTraining finished. Total time elapsed: {time.strftime("%H:%M:%S", time.gmtime(time_elapsed))}')
