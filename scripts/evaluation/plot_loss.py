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
        print("Parsing log file...")
        for line in file:
            step = extract_step_from_line(line)
            loss = extract_loss_from_line(line)
            if step not in steps and step is not None:
                if loss is not None:
                    losses.append(loss)
                    steps.append(step)
        print("Log file parsed")
    return steps, losses

def plot_loss_from_file(file_path):
    steps, losses = get_steps_and_losses(file_path)

    # Plot the first figure
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2 rows, 1 column
    ax1.plot(steps, losses, label='Step Loss')
    ax1.set_title('Training Loss Over Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Smooth the losses
    window_size = 100
    smoothed_losses = moving_average(losses, window_size)

    # Plot the second figure
    ax2.plot(steps[:len(smoothed_losses)], smoothed_losses, label=f'Moving Average (Window Size: {window_size})', color='red')

    # Plot tendency line for the moving average
    tendency_line = np.polyfit(steps[:len(smoothed_losses)], smoothed_losses, 1)
    steepness = tendency_line[0]  # Slope of the line
    ax2.plot(steps[:len(smoothed_losses)], np.polyval(tendency_line, steps[:len(smoothed_losses)]), label=f'Tendency Line (Steepness: {steepness:.4e})', linestyle='dashed', color='green')

    ax2.set_title('Training Loss Over Steps with Moving Average')
    ax2.set_xlabel('Steps (%)')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Add filename to the bottom of the plot
    fig.text(0.5, 0.01, f"Filename: {args.filename}", ha='center', fontsize=8)

    # Adjust layout to prevent clipping of titles and labels
    plt.tight_layout()

    # Save the combined figure
    output_filename = f"./../../output/plots/combined_plots_{file_path.split('/')[-1].split('.')[-2]}.png"
    fig.savefig(output_filename)
    plt.show()

    print(f"Combined plot saved at {output_filename}")

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
    plot_loss_from_file(f"./../../model_logs/{args.filename}")
