import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from main.model.gat import GAT, SpGAT, GatedResidualDiffusionGAT
from main.model.model import Decoder, DiffusionPropagate, Encoder, GNNModel, VAEModel
from main.utils import InverseProblemDataset, adj_process, diffusion_evaluation, load_dataset


MODEL_NAME = "DeepIM-GRDGAT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA_LAUNCH_BLOCKING = 1

parser = argparse.ArgumentParser(description=MODEL_NAME)
datasets = ["jazz", "cora_ml", "power_grid", "netscience", "random5"]
parser.add_argument(
    "-d",
    "--dataset",
    default="cora_ml",
    type=str,
    help="one of: {}".format(", ".join(sorted(datasets))),
)
diffusion = ["IC", "LT", "SIS"]
parser.add_argument(
    "-dm",
    "--diffusion_model",
    default="LT",
    type=str,
    help="one of: {}".format(", ".join(sorted(diffusion))),
)
seed_rate = [1, 5, 10, 20]
parser.add_argument(
    "-sp",
    "--seed_rate",
    default=1,
    type=int,
    help="one of: {}".format(", ".join(str(sorted(seed_rate)))),
)
mode = ["Normal", "Budget Constraint"]
parser.add_argument(
    "-m",
    "--mode",
    default="normal",
    type=str,
    help="one of: {}".format(", ".join(sorted(mode))),
)
args = parser.parse_args()


def print_section(title):
    line = "=" * 72
    print("\n{}".format(line))
    print(title)
    print(line)


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                continue

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                continue


def setup_experiment_logger(model_name, dataset, diffusion_model, seed_rate):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (
        f"{model_name}_{dataset}_{diffusion_model}_seed{seed_rate * 10}_{timestamp}.txt"
    )
    log_file = open(log_path, "w", encoding="utf-8")
    tee_logger = TeeLogger(sys.__stdout__, log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    print("Log file: {}".format(log_path.resolve()))
    return log_file, log_path, original_stdout, original_stderr


def close_experiment_logger(log_file, original_stdout, original_stderr):
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def print_config(title, config):
    print_section(title)
    for key, value in config.items():
        print("{:<24}: {}".format(key, value))


def log_epoch_metrics(epoch, total_epochs, metrics):
    print(
        "[Train][Epoch {}/{}] total_loss={:.4f} recon_loss={:.4f} "
        "forward_loss={:.4f} recon_precision={:.4f} recon_recall={:.4f} "
        "time={:.4f}s".format(
            epoch,
            total_epochs,
            metrics["total_loss"],
            metrics["recon_loss"],
            metrics["forward_loss"],
            metrics["recon_precision"],
            metrics["recon_recall"],
            metrics["epoch_time"],
        )
    )


def log_inverse_metrics(
    step, total_steps, total_loss, spread_loss, budget_loss, binary_loss, seed_mass
):
    print(
        "[Inverse][Step {}/{}] total_loss={:.5f} spread_loss={:.5f} "
        "budget_loss={:.5f} binary_loss={:.5f} seed_mass={:.2f}".format(
            step,
            total_steps,
            total_loss,
            spread_loss,
            budget_loss,
            binary_loss,
            seed_mass,
        )
    )


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sampling(inverse_pairs):
    diffusion_count = []
    for pair in inverse_pairs:
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1 * inverse_pairs.shape[0])).indices
    return top_k


def build_diffusion_helpers(adj_matrix):
    graph = nx.from_scipy_sparse_array(adj_matrix)
    neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    degrees = {node: max(1, graph.degree(node)) for node in graph.nodes()}
    lt_thresholds = {
        node: max(1, int(np.ceil(0.5 * graph.degree(node))))
        for node in graph.nodes()
    }
    return graph, neighbors, degrees, lt_thresholds


def simulate_ic(seeds, neighbors, degrees, rng):
    active = set(int(node) for node in seeds)
    frontier = list(active)

    while frontier:
        new_frontier = []
        for node in frontier:
            for nbr in neighbors[node]:
                if nbr in active:
                    continue
                if rng.random() < 1.0 / degrees[nbr]:
                    active.add(nbr)
                    new_frontier.append(nbr)
        frontier = new_frontier

    return len(active)


def simulate_lt(seeds, neighbors, lt_thresholds):
    active = set(int(node) for node in seeds)
    frontier = list(active)
    active_neighbor_count = defaultdict(int)

    while frontier:
        new_frontier = []
        for node in frontier:
            for nbr in neighbors[node]:
                if nbr in active:
                    continue
                active_neighbor_count[nbr] += 1
                if active_neighbor_count[nbr] >= lt_thresholds[nbr]:
                    active.add(nbr)
                    new_frontier.append(nbr)
        frontier = new_frontier

    return len(active)


def estimate_spread_fast(
    seeds, diffusion_model, neighbors, degrees, lt_thresholds, mc_runs=50, base_seed=42
):
    spreads = []
    for offset in range(mc_runs):
        rng = np.random.default_rng(base_seed + offset)
        if diffusion_model == "IC":
            spread = simulate_ic(seeds, neighbors, degrees, rng)
        else:
            spread = simulate_lt(seeds, neighbors, lt_thresholds)
        spreads.append(spread)
    return float(np.mean(spreads))


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)


def greedy_refine_seed_set(
    candidate_scores,
    adj_matrix,
    diffusion_model,
    budget,
    pool_size=50,
    mc_runs=50,
    random_seed=42,
):
    graph, neighbors, degrees, lt_thresholds = build_diffusion_helpers(adj_matrix)
    pagerank_scores = nx.pagerank(graph)
    pagerank_array = np.array(
        [pagerank_scores[node] for node in graph.nodes()], dtype=np.float32
    )

    combined_scores = 0.8 * normalize_scores(candidate_scores) + 0.2 * normalize_scores(
        pagerank_array
    )
    pool_size = min(graph.number_of_nodes(), max(pool_size, budget))
    candidate_pool = np.argsort(-combined_scores)[:pool_size].tolist()

    selected = []
    current_spread = 0.0
    for step in range(budget):
        best_node = None
        best_spread = -1.0
        for node in candidate_pool:
            if node in selected:
                continue
            spread = estimate_spread_fast(
                selected + [node],
                diffusion_model,
                neighbors,
                degrees,
                lt_thresholds,
                mc_runs=mc_runs,
                base_seed=random_seed + step * 9973 + node,
            )
            if spread > best_spread:
                best_spread = spread
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)
        current_spread = best_spread

    return selected, current_spread


log_file, log_path, original_stdout, original_stderr = setup_experiment_logger(
    MODEL_NAME, args.dataset, args.diffusion_model, args.seed_rate
)

with open(
    "data/" + args.dataset + "_mean_" + args.diffusion_model + str(10 * args.seed_rate) + ".SG",
    "rb",
) as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph["adj"], graph["inverse_pairs"]

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse()

if args.dataset == "random5":
    batch_size = 2
    hidden_dim = 4096
    latent_dim = 1024
else:
    batch_size = 16
    hidden_dim = 1024
    latent_dim = 512

num_epochs = 20
inverse_steps = 10
learning_rate = 1e-4
forward_hidden_dim = 64
forward_heads = 4
forward_dropout = 0.2
forward_alpha = 0.2

train_set, test_set = torch.utils.data.random_split(
    inverse_pairs, [len(inverse_pairs) - batch_size, batch_size]
)

train_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False
)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

encoder = Encoder(
    input_dim=inverse_pairs.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim
)

decoder = Decoder(
    input_dim=latent_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    output_dim=inverse_pairs.shape[1],
)

vae_model = VAEModel(Encoder=encoder, Decoder=decoder).to(device)

forward_model = GatedResidualDiffusionGAT(
    nfeat=1,
    nhid=forward_hidden_dim,
    nclass=1,
    dropout=forward_dropout,
    nheads=forward_heads,
    alpha=forward_alpha,
)

optimizer = Adam(
    [{"params": vae_model.parameters()}, {"params": forward_model.parameters()}],
    lr=learning_rate,
)

experiment_config = {
    "model": MODEL_NAME,
    "dataset": args.dataset,
    "diffusion_model": args.diffusion_model,
    "seed_rate_percent": args.seed_rate * 10,
    "device": device,
    "num_nodes": inverse_pairs.shape[1],
    "num_samples": len(inverse_pairs),
    "train_samples": len(train_set),
    "test_samples": len(test_set),
    "batch_size": batch_size,
    "encoder_hidden_dim": hidden_dim,
    "latent_dim": latent_dim,
    "forward_hidden_dim": forward_hidden_dim,
    "forward_heads": forward_heads,
    "dropout": forward_dropout,
    "alpha": forward_alpha,
    "learning_rate": learning_rate,
    "train_epochs": num_epochs,
    "inverse_steps": inverse_steps,
    "refine_pool_size": 50,
    "refine_mc_runs": 50,
}
print_config("Experiment Configuration", experiment_config)

adj = adj.to(device)
forward_model = forward_model.to(device)
forward_model.train()


def loss_all(x, x_hat, y, y_hat):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
    forward_loss = F.mse_loss(y_hat, y, reduction="sum")
    return reproduction_loss + forward_loss, reproduction_loss, forward_loss


print_section("Stage 1: Joint Training")

for epoch in range(num_epochs):
    begin = time.time()
    total_overall = 0
    forward_loss_total = 0
    reproduction_loss_total = 0
    precision_re = 0
    recall_re = 0

    for data_pair in train_loader:
        x = data_pair[:, :, 0].float().to(device)
        y = data_pair[:, :, 1].float().to(device)
        optimizer.zero_grad()

        x_true = x.cpu().detach().numpy()
        loss = 0
        for i, x_i in enumerate(x):
            y_i = y[i]

            x_hat = vae_model(x_i.unsqueeze(0))
            y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj)
            total, re, forw = loss_all(x_i.unsqueeze(0), x_hat, y_i, y_hat.squeeze(-1))

            loss += total
            reproduction_loss_total += re.item()
            forward_loss_total += forw.item()

            x_pred = x_hat.cpu().detach().numpy()
            x_pred[x_pred > 0.01] = 1
            x_pred[x_pred != 1] = 0

            precision_re += precision_score(x_true[i], x_pred[0], zero_division=0)
            recall_re += recall_score(x_true[i], x_pred[0], zero_division=0)

        total_overall += loss.item()
        loss = loss / x.size(0)

        loss.backward()
        optimizer.step()
        for p in forward_model.parameters():
            p.data.clamp_(min=0)

    end = time.time()
    log_epoch_metrics(
        epoch + 1,
        num_epochs,
        {
            "total_loss": total_overall / len(train_set),
            "recon_loss": reproduction_loss_total / len(train_set),
            "forward_loss": forward_loss_total / len(train_set),
            "recon_precision": precision_re / len(train_set),
            "recon_recall": recall_re / len(train_set),
            "epoch_time": end - begin,
        },
    )

for param in vae_model.parameters():
    param.requires_grad = False

for param in forward_model.parameters():
    param.requires_grad = False

encoder = vae_model.Encoder
decoder = vae_model.Decoder


def loss_inverse(y_hat, x_hat, target_seed_count):
    spread_loss = -torch.sigmoid(y_hat).mean()
    seed_mass = x_hat.sum()
    budget_loss = ((seed_mass - target_seed_count) ** 2) / x_hat.shape[1]
    binary_loss = torch.mean(x_hat * (1.0 - x_hat))
    total_loss = spread_loss + 5.0 * budget_loss + 0.1 * binary_loss
    return total_loss, spread_loss, budget_loss, binary_loss, seed_mass


with open(
    "data/" + args.dataset + "_mean_" + args.diffusion_model + str(10 * args.seed_rate) + ".SG",
    "rb",
) as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph["adj"], graph["inverse_pairs"]

adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = torch.Tensor(adj.toarray()).to_sparse().to(device)

topk_seed = sampling(inverse_pairs)

z_hat = 0
for i in topk_seed:
    z_hat += encoder(inverse_pairs[i, :, 0].unsqueeze(0).to(device))

z_hat = z_hat / len(topk_seed)
# Use the user-specified percentage budget directly.
# For example, jazz has 198 nodes, so 1% maps to 1 seed node.
seed_num = int(x_hat.sum().item())

z_hat = z_hat.detach()
z_hat.requires_grad = True
z_optimizer = Adam([z_hat], lr=learning_rate)

print_section("Stage 2: Inverse Seed Optimization")

for i in range(inverse_steps):
    z_optimizer.zero_grad()
    x_hat = decoder(z_hat)
    y_hat = forward_model(x_hat.squeeze(0).unsqueeze(-1), adj).squeeze(-1).unsqueeze(0)

    loss, spread_loss, budget_loss, binary_loss, seed_mass = loss_inverse(
        y_hat, x_hat, seed_num
    )

    loss.backward()
    z_optimizer.step()

    log_inverse_metrics(
        i + 1,
        inverse_steps,
        loss.item(),
        spread_loss.item(),
        budget_loss.item(),
        binary_loss.item(),
        seed_mass.item(),
    )

candidate_scores = x_hat.squeeze(0).detach().cpu().numpy()
refine_pool_size = min(inverse_pairs.shape[1], max(seed_num * 20, 50))
seed, fast_spread_score = greedy_refine_seed_set(
    candidate_scores,
    graph["adj"],
    args.diffusion_model,
    seed_num,
    pool_size=refine_pool_size,
    mc_runs=50,
    random_seed=42,
)
seed = np.asarray(seed, dtype=int)

with open(
    "data/" + args.dataset + "_mean_" + args.diffusion_model + str(10 * args.seed_rate) + ".SG",
    "rb",
) as f:
    graph = pickle.load(f)

adj, inverse_pairs = graph["adj"], graph["inverse_pairs"]

influence = diffusion_evaluation(adj, seed, diffusion=args.diffusion_model)
print_config(
    "Final Evaluation",
    {
        "model": MODEL_NAME,
        "dataset": args.dataset,
        "diffusion_model": args.diffusion_model,
        "selected_seed_count": len(seed),
        "predicted_seed_indices": seed.tolist(),
        "fast_refine_spread": round(fast_spread_score, 4),
        "diffusion_count": influence,
    },
)
print("\nExperiment log saved to: {}".format(log_path.resolve()))
close_experiment_logger(log_file, original_stdout, original_stderr)
