import matplotlib.pyplot as plt

def plot_gate_distribution(model, save_path):

    gates = []

    for g in model.get_all_gates():
        gates.extend(g.detach().cpu().numpy().flatten())

    plt.figure(figsize=(6, 4))
    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    plt.savefig(save_path)
    plt.close()