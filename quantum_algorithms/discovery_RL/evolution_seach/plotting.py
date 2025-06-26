import csv
import matplotlib.pyplot as plt

def plot_fidelity_vs_depth(csv_path, title="Evolution Progress"):
    generations = []
    fidelities = []
    depths = []

    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            generations.append(int(row["generation"]))
            fidelities.append(float(row["fidelity"]))
            depths.append(int(row["depth"]))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fidelity", color=color)
    ax1.plot(generations, fidelities, color=color, label="Fidelity")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("Depth", color=color)
    ax2.plot(generations, depths, color=color, linestyle='dashed', label="Depth")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(title)
    plt.savefig("output/plot.png")
    plt.show()
