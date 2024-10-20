import os
import re
import sys

import matplotlib.pyplot as plt


def parse_log_file(file_path):
    steps = []
    val_losses = []
    train_times = []

    with open(file_path, "r") as f:
        for line in f:
            if "val_loss" in line:
                match = re.search(
                    r"step:(\d+)/\d+ val_loss:([\d.]+).*train_time:([\d.]+)ms", line
                )
                if match:
                    step = int(match.group(1))
                    val_loss = float(match.group(2))
                    train_time = float(match.group(3)) / 1000  # Convert ms to seconds
                    steps.append(step)
                    val_losses.append(val_loss)
                    train_times.append(train_time)

    return steps, val_losses, train_times


def plot_val_loss_vs_tokens(data, y_min=None, y_max=None):
    plt.figure(figsize=(12, 7))
    for filename, (tokens, val_losses) in data.items():
        plt.plot(tokens, val_losses, marker="o", label=filename)

    plt.title("Validation Loss vs. Number of Training Tokens")
    plt.xlabel("Number of Training Tokens")
    plt.ylabel("Validation Loss")
    plt.xscale("log")

    plt.xlim(
        min(min(tokens) for tokens in [d[0] for d in data.values()]),
        max(max(tokens) for tokens in [d[0] for d in data.values()]),
    )

    if y_min is None or y_max is None:
        all_losses = [loss for _, losses in data.values() for loss in losses]
        data_y_min = min(all_losses)
        data_y_max = max(all_losses)
        y_range = data_y_max - data_y_min
        y_min = data_y_min - 0.02 * y_range if y_min is None else y_min
        y_max = data_y_max + 0.02 * y_range if y_max is None else y_max

    plt.ylim(y_min, y_max)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_loss_vs_tokens_plot.png")
    plt.close()


def plot_val_loss_vs_time(data, y_min=None, y_max=None):
    plt.figure(figsize=(12, 7))
    for filename, (train_times, val_losses) in data.items():
        plt.plot(train_times, val_losses, marker="o", label=filename)

    plt.title("Validation Loss vs. Wall Clock Time")
    plt.xlabel("Wall Clock Time (seconds)")
    plt.ylabel("Validation Loss")

    plt.xlim(0, max(max(times) for times in [d[0] for d in data.values()]))

    if y_min is None or y_max is None:
        all_losses = [loss for _, losses in data.values() for loss in losses]
        data_y_min = min(all_losses)
        data_y_max = max(all_losses)
        y_range = data_y_max - data_y_min
        y_min = data_y_min - 0.02 * y_range if y_min is None else y_min
        y_max = data_y_max + 0.02 * y_range if y_max is None else y_max

    plt.ylim(y_min, y_max)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_loss_vs_time_plot.png")
    plt.close()


# Main execution
logs_folder = "logs"  # Replace with the actual path to your logs folder
data = {}

for filename in os.listdir(logs_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(logs_folder, filename)
        steps, val_losses, train_times = parse_log_file(file_path)

        # Calculate tokens based on steps
        tokens_per_step = 8 * 64 * 1024  # batch_size * sequence_length
        tokens = [step * tokens_per_step for step in steps]

        data[filename] = (tokens, val_losses)

# Plot loss vs tokens
plot_val_loss_vs_tokens(data)

# Update data dictionary for time plot
time_data = {
    filename: (train_times, val_losses) for filename, (_, val_losses) in data.items()
}

# Plot loss vs wall clock time
plot_val_loss_vs_time(time_data)

print(
    "Plots have been saved as 'val_loss_vs_tokens_plot.png' and 'val_loss_vs_time_plot.png'"
)
