import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_loss_from_line(line):
    match = re.search(r'step_loss=([\d.]+)', line)
    if match:
        return float(match.group(1))
    return None

def extract_step_from_line(line):
    match = re.search(r'\b(\d+)\b/\d+', line)
    if match:
        return int(match.group(1))
    return None

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_steps_and_losses(file_path):
    losses = []
    steps = []
    with open(file_path, 'r') as file:
        for line in file:
            step = extract_step_from_line(line)
            loss = extract_loss_from_line(line)
            if step not in steps and step is not None:
                if loss is not None:
                    losses.append(loss)
                    steps.append(step)
    return steps, losses

def plot_loss_from_file(file_path):

    steps, losses = get_steps_and_losses(file_path)

    plt.plot(steps, losses, label='Step Loss')
    plt.title('Training Loss Over Steps')
    plt.xlabel('Steps (%)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    window_size = 100
    smoothed_losses = moving_average(losses, window_size)

    # Plot smoothed loss
    smoothed_steps = steps[:len(smoothed_losses)] 
    plt.plot(smoothed_steps, smoothed_losses, label=f'Moving Average (Window Size: {window_size})', color='red')
    
    # Plot tendency line for the moving average
    tendency_line = np.polyfit(smoothed_steps, smoothed_losses, 1)
    plt.plot(smoothed_steps, np.polyval(tendency_line, smoothed_steps), label='Tendency Line', linestyle='dashed', color='green')

    plt.title('Training Loss Over Steps with Moving Average')
    plt.xlabel('Steps (%)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description ='Plot the loss of a training given the log file')
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=True,
        help="Filename of the log file inside `model_logs`.",
    )
    return parser.parse_args()


if __name__=="__main__": 
    args = parse_args()
    plot_loss_from_file(f"./../model_logs/{args.filename}")
