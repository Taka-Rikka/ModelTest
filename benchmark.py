import argparse
import ast
import heapq
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark IM methods on the same dataset and diffusion setup."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="jazz",
        choices=["jazz", "cora_ml", "power_grid", "netscience", "random5"],
        help="Dataset name.",
    )
    parser.add_argument(
        "-dm",
        "--diffusion_model",
        default="LT",
        choices=["LT", "IC"],
        help="Diffusion model used for seed selection and evaluation.",
    )
    parser.add_argument(
        "-sp",
        "--seed_rate",
        default=1,
        type=int,
        choices=[1, 5, 10, 20],
        help="Seed count used by benchmark seed selection, e.g. 1 means select exactly 1 seed node.",
    )
    parser.add_argument(
        "--methods",
        default="Degree,PageRank,IMM,DeepIM-GRDGAT,CELF,CELF++",
        help="Comma-separated method list.",
    )
    parser.add_argument(
        "--selection_sims",
        default=50,
        type=int,
        help="Monte Carlo runs used inside CELF/CELF++.",
    )
    parser.add_argument(
        "--eval_sims",
        default=200,
        type=int,
        help="Monte Carlo runs used for the final comparison table.",
    )
    parser.add_argument(
        "--rr_sets",
        default=1000,
        type=int,
        help="Number of RR sets used by IMM.",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Random seed for the benchmark.",
    )
    parser.add_argument(
        "--deepim_script",
        default="genim.py",
        help="Path to the DeepIM-GRDGAT entry script.",
    )
    parser.add_argument(
        "--deepim_timeout",
        default=7200,
        type=int,
        help="Timeout in seconds for DeepIM-GRDGAT training.",
    )
    parser.add_argument(
        "--deepim_log",
        default="",
        help="Optional existing DeepIM-GRDGAT log file. If set, benchmark.py parses seeds from it instead of rerunning genim.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="benchmarks",
        help="Directory where csv results will be written.",
    )
    return parser.parse_args()


def load_graph_data(dataset, diffusion_model, seed_rate, data_dir="data"):
    data_path = Path(data_dir) / f"{dataset}_mean_{diffusion_model}{seed_rate * 10}.SG"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    sys.path.append(str(Path(data_dir).resolve()))
    with data_path.open("rb") as f:
        graph = pickle.load(f)

    adj = graph["adj"].tocsr()
    inverse_pairs = graph["inverse_pairs"]
    return adj, inverse_pairs, data_path


def compute_seed_budget(num_nodes, seed_rate):
    return max(1, min(num_nodes, int(seed_rate)))


def build_graph_bundle(adj):
    graph = nx.from_scipy_sparse_array(adj)
    neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    degrees = {node: max(1, graph.degree(node)) for node in graph.nodes()}
    lt_thresholds = {node: max(1, int(np.ceil(0.5 * graph.degree(node)))) for node in graph.nodes()}
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


def estimate_spread(seeds, diffusion_model, neighbors, degrees, lt_thresholds, mc_runs, base_seed):
    spreads = []
    for offset in range(mc_runs):
        rng = np.random.default_rng(base_seed + offset)
        if diffusion_model == "IC":
            spread = simulate_ic(seeds, neighbors, degrees, rng)
        elif diffusion_model == "LT":
            spread = simulate_lt(seeds, neighbors, lt_thresholds)
        else:
            raise ValueError(f"Unsupported diffusion model: {diffusion_model}")
        spreads.append(spread)

    spread_array = np.asarray(spreads, dtype=float)
    return float(spread_array.mean()), float(spread_array.std(ddof=0))


def make_cached_spread_estimator(diffusion_model, neighbors, degrees, lt_thresholds, mc_runs, random_seed):
    cache = {}

    def estimate(seeds):
        key = tuple(sorted(int(node) for node in seeds))
        if key not in cache:
            cache[key] = estimate_spread(
                key,
                diffusion_model,
                neighbors,
                degrees,
                lt_thresholds,
                mc_runs,
                random_seed + len(cache) * 9973,
            )[0]
        return cache[key]

    return estimate


def select_degree(graph, k):
    ranked = sorted(graph.degree, key=lambda item: (-item[1], item[0]))
    return [node for node, _ in ranked[:k]]


def select_pagerank(graph, k):
    scores = nx.pagerank(graph)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [node for node, _ in ranked[:k]]


def select_celf(graph, k, estimate):
    selected = []
    current_spread = 0.0
    heap = []

    for node in graph.nodes():
        gain = estimate([node])
        heapq.heappush(heap, (-gain, node, 0))

    while len(selected) < k and heap:
        neg_gain, node, flag = heapq.heappop(heap)
        if node in selected:
            continue
        if flag == len(selected):
            selected.append(node)
            current_spread = estimate(selected)
            continue

        gain = estimate(selected + [node]) - current_spread
        heapq.heappush(heap, (-gain, node, len(selected)))

    return selected


def select_celfpp(graph, k, estimate):
    selected = []
    current_spread = 0.0
    last_seed = None
    heap = []

    for node in graph.nodes():
        gain = estimate([node])
        heapq.heappush(heap, (-gain, node, 0, None, gain))

    while len(selected) < k and heap:
        neg_gain, node, flag, prev_best, mg2 = heapq.heappop(heap)
        if node in selected:
            continue

        if flag == len(selected):
            selected.append(node)
            current_spread = estimate(selected)
            last_seed = node
            continue

        if prev_best is not None and last_seed is not None and prev_best == last_seed:
            gain = mg2
        else:
            gain = estimate(selected + [node]) - current_spread
            if last_seed is not None and node != last_seed:
                seed_with_last = list(selected)
                if last_seed not in seed_with_last:
                    seed_with_last.append(last_seed)
                gain2 = estimate(seed_with_last + [node]) - estimate(seed_with_last)
            else:
                gain2 = gain
            prev_best = last_seed
            mg2 = gain2

        heapq.heappush(heap, (-gain, node, len(selected), prev_best, mg2))

    return selected


def generate_rr_set_ic(start, reverse_neighbors, degrees, rng):
    rr_set = {start}
    frontier = [start]

    while frontier:
        node = frontier.pop()
        for parent in reverse_neighbors[node]:
            if parent in rr_set:
                continue
            if rng.random() < 1.0 / degrees[node]:
                rr_set.add(parent)
                frontier.append(parent)

    return rr_set


def generate_rr_set_lt(start, reverse_neighbors, rng):
    # This is a practical LT-style reverse live-edge approximation.
    rr_set = {start}
    frontier = [start]

    while frontier:
        node = frontier.pop()
        parents = reverse_neighbors[node]
        if not parents:
            continue
        chosen_parent = parents[int(rng.integers(0, len(parents)))]
        if chosen_parent not in rr_set:
            rr_set.add(chosen_parent)
            frontier.append(chosen_parent)

    return rr_set


def select_imm(graph, k, diffusion_model, rr_sets_num, random_seed):
    reverse_neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()}
    degrees = {node: max(1, graph.degree(node)) for node in graph.nodes()}
    nodes = list(graph.nodes())
    rng = np.random.default_rng(random_seed)

    rr_sets = []
    node_to_rr = defaultdict(set)
    for rr_idx in range(rr_sets_num):
        start = int(rng.choice(nodes))
        if diffusion_model == "IC":
            rr_set = generate_rr_set_ic(start, reverse_neighbors, degrees, rng)
        elif diffusion_model == "LT":
            rr_set = generate_rr_set_lt(start, reverse_neighbors, rng)
        else:
            raise ValueError(f"IMM only supports IC/LT here, got {diffusion_model}")

        rr_sets.append(rr_set)
        for node in rr_set:
            node_to_rr[node].add(rr_idx)

    gains = {node: len(node_to_rr[node]) for node in graph.nodes()}
    covered = set()
    selected = []

    for _ in range(k):
        best_node = None
        best_gain = -1
        for node in graph.nodes():
            if node in selected:
                continue
            gain = gains[node]
            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)
        newly_covered = node_to_rr[best_node] - covered
        covered.update(newly_covered)

        for rr_idx in newly_covered:
            for node in rr_sets[rr_idx]:
                if node not in selected:
                    gains[node] -= 1

    return selected


def parse_deepim_log(log_text):
    def parse_field(name, cast=None, default=None):
        match = re.search(rf"{re.escape(name)}\s*:\s*([^\n]+)", log_text)
        if not match:
            return default
        value = match.group(1).strip()
        return cast(value) if cast is not None else value

    seed_text = parse_field("predicted_seed_indices")
    if seed_text is None:
        raise ValueError("Failed to parse predicted_seed_indices from DeepIM log.")

    return {
        "model": parse_field("model"),
        "dataset": parse_field("dataset"),
        "diffusion_model": parse_field("diffusion_model"),
        "selected_seed_count": parse_field("selected_seed_count", int),
        "predicted_seed_indices": list(ast.literal_eval(seed_text)),
        "diffusion_count": parse_field("diffusion_count", float),
    }


def run_deepim_method(dataset, diffusion_model, seed_rate, deepim_script, timeout, deepim_log=""):
    if deepim_log:
        log_text = Path(deepim_log).read_text(encoding="utf-8", errors="ignore")
        parsed_log = parse_deepim_log(log_text)
        if parsed_log["dataset"] and parsed_log["dataset"] != dataset:
            raise ValueError(
                f"DeepIM log dataset mismatch: log has {parsed_log['dataset']}, "
                f"benchmark expects {dataset}."
            )
        if parsed_log["diffusion_model"] and parsed_log["diffusion_model"] != diffusion_model:
            raise ValueError(
                f"DeepIM log diffusion mismatch: log has {parsed_log['diffusion_model']}, "
                f"benchmark expects {diffusion_model}."
            )
        return parsed_log["predicted_seed_indices"]

    command = [
        sys.executable,
        deepim_script,
        "-d",
        dataset,
        "-dm",
        diffusion_model,
        "-sp",
        str(seed_rate),
    ]
    process = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    output = process.stdout + "\n" + process.stderr
    if process.returncode != 0:
        raise RuntimeError(f"DeepIM-GRDGAT failed.\n{output}")

    return parse_deepim_log(output)["predicted_seed_indices"]


def benchmark_method(
    method_name,
    graph,
    k,
    diffusion_model,
    neighbors,
    degrees,
    lt_thresholds,
    args,
):
    start = time.perf_counter()

    if method_name == "Degree":
        seeds = select_degree(graph, k)
    elif method_name == "PageRank":
        seeds = select_pagerank(graph, k)
    elif method_name == "CELF":
        estimator = make_cached_spread_estimator(
            diffusion_model,
            neighbors,
            degrees,
            lt_thresholds,
            args.selection_sims,
            args.random_seed,
        )
        seeds = select_celf(graph, k, estimator)
    elif method_name == "CELF++":
        estimator = make_cached_spread_estimator(
            diffusion_model,
            neighbors,
            degrees,
            lt_thresholds,
            args.selection_sims,
            args.random_seed,
        )
        seeds = select_celfpp(graph, k, estimator)
    elif method_name == "IMM":
        seeds = select_imm(graph, k, diffusion_model, args.rr_sets, args.random_seed)
    elif method_name == "DeepIM-GRDGAT":
        seeds = run_deepim_method(
            args.dataset,
            diffusion_model,
            args.seed_rate,
            args.deepim_script,
            args.deepim_timeout,
            args.deepim_log,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    selection_time = time.perf_counter() - start
    eval_start = time.perf_counter()
    mean_spread, std_spread = estimate_spread(
        seeds,
        diffusion_model,
        neighbors,
        degrees,
        lt_thresholds,
        args.eval_sims,
        args.random_seed + 100000,
    )
    eval_time = time.perf_counter() - eval_start

    return {
        "method": method_name,
        "dataset": args.dataset,
        "diffusion_model": diffusion_model,
        "seed_rate_percent": args.seed_rate,
        "seed_budget": k,
        "selected_seed_count": len(seeds),
        "mean_spread": round(mean_spread, 4),
        "std_spread": round(std_spread, 4),
        "selection_time_sec": round(selection_time, 4),
        "evaluation_time_sec": round(eval_time, 4),
        "seeds": seeds,
    }


def main():
    args = parse_args()
    method_names = [name.strip() for name in args.methods.split(",") if name.strip()]

    adj, inverse_pairs, data_path = load_graph_data(
        args.dataset,
        args.diffusion_model,
        args.seed_rate,
    )
    graph, neighbors, degrees, lt_thresholds = build_graph_bundle(adj)
    k = compute_seed_budget(adj.shape[0], args.seed_rate)

    print(f"Dataset file: {data_path}")
    print(
        f"Nodes: {adj.shape[0]}, Edges: {graph.number_of_edges()}, "
        f"Seed count: {args.seed_rate}, Seed budget: {k}"
    )
    print(f"Methods: {', '.join(method_names)}")

    results = []
    for method_name in method_names:
        print(f"\nRunning {method_name} ...")
        result = benchmark_method(
            method_name,
            graph,
            k,
            args.diffusion_model,
            neighbors,
            degrees,
            lt_thresholds,
            args,
        )
        results.append(result)
        print(
            f"{method_name}: mean_spread={result['mean_spread']}, "
            f"std={result['std_spread']}, seeds={result['selected_seed_count']}"
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["mean_spread", "std_spread"],
        ascending=[False, True],
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / (
        f"benchmark_{args.dataset}_{args.diffusion_model}_seedcount{args.seed_rate}_{timestamp}.csv"
    )
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nBenchmark Results")
    print(result_df.to_string(index=False))
    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
