import time
import sys


class ProgressBar:
    """
    A progress bar for tracking training progress.

    This class provides a simple progress bar that can be used to track the
    progress of a training loop.

    Attributes:
        total_steps (int): The total number of steps in an epoch.
        epochs (int): The total number of epochs.
        bar_length (int): The length of the progress bar.
        start_time (float): The time at which training started.
    """

    def __init__(self, total_steps, epochs, bar_length=50):
        """
        Initializes the ProgressBar.

        Args:
            total_steps: The total number of steps in an epoch.
            epochs: The total number of epochs.
            bar_length: The length of the progress bar.
        """
        self.total_steps = total_steps
        self.epochs = epochs
        self.bar_length = bar_length
        self.start_time = None

    def start(self):
        """
        Starts the progress bar.
        """
        self.start_time = time.time()

    def update(self, epoch, current_step, metrics):
        """
        Updates the progress bar.

        Args:
            epoch: The current epoch.
            current_step: The current step in the epoch.
            metrics: A dictionary of metrics to display.
        """
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
            f"\rEpoch {epoch + 1}/{self.epochs} [{current_step}/{self.total_steps}] "
            f"[{bar}] {percent * 100:.1f}% - "
            f"ETA: {time.strftime('%H:%M:%S', time.gmtime(time_remaining))} | "
            f"{metrics_str}"
        )
        sys.stdout.flush()

    def end(self):
        """
        Ends the progress bar.
        """
        time_elapsed = time.time() - self.start_time
        print(f'\n\nTraining finished. Total time elapsed: {time.strftime("%H:%M:%S", time.gmtime(time_elapsed))}')