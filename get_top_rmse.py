import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
def calculate_average_and_variance(base_dir, k):
    np.set_printoptions(precision=10, suppress=True)

    results_by_experiment = defaultdict(list)
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "results_eval_metrics.json":
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        
                        experiment_name = os.path.basename(root).split("_", 1)[1]
                        
                        seed = data.get("random_seed")

                        
                        pearson = data.get("pearson_correlation")
                        spearman = data.get("spearman_correlation")
                        rmse = data.get("rmse") 
                        
                        if pearson is not None and spearman is not None and seed is not None:
                            results_by_experiment[experiment_name].append({
                                "seed": seed,
                                "pearson": pearson,
                                "spearman": spearman,
                                "rmse": rmse
                            })
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error reading {json_path}: {e}")


    summary = []
    top_seeds_set = set()

    for experiment_name, results in results_by_experiment.items():

        sorted_results = sorted(results, key=lambda x: x["rmse"], reverse=False)


        top_3_results = sorted_results[:k]

        top_3_pearson = [result["pearson"] for result in top_3_results]
        top_3_spearman = [result["spearman"] for result in top_3_results]
        top_3_rmse = [result["rmse"] for result in top_3_results]
        
        pearson_mean = np.mean(top_3_pearson)
        spearman_mean = np.mean(top_3_spearman)
        rmse_mean = np.mean(top_3_rmse) 
        
        pearson_variance = np.var(top_3_pearson, ddof=0)
        spearman_variance = np.var(top_3_spearman, ddof=0)
        rmse_variance = np.var(top_3_rmse, ddof=0)
        
        top_3_seeds = [result["seed"] for result in top_3_results]
        top_seeds_set.update(top_3_seeds)

        summary.append({
            "Experiment": experiment_name,
            "Seed number": len(top_3_results),
            "Top 3 Seeds": ', '.join(map(str, top_3_seeds)),
            "Pearson mean": pearson_mean,
            "Pearson variance": pearson_variance,
            "Spearman mean": spearman_mean,
            "Spearman variance": spearman_variance,
            "RMSE mean": rmse_mean, 
            "RMSE variance": rmse_variance 
        })

    pd.set_option('display.max_colwidth', None)

    df = pd.DataFrame(summary)


    df["Pearson mean"] = df["Pearson mean"].apply(lambda x: f"{x:.3f}")
    df["Spearman mean"] = df["Spearman mean"].apply(lambda x: f"{x:.3f}")
    df["RMSE mean"] = df["RMSE mean"].apply(lambda x: f"{x:.3f}")
    pd.set_option('display.float_format', '{:,.3e}'.format) 


    df["Contains baseline"] = df["Experiment"].str.contains("baseline")
    df["Name length"] = df["Experiment"].str.len()

    df = df.sort_values(by=["Contains baseline", "Name length"], ascending=[False, True])


    df = df.drop(columns=["Contains baseline", "Name length"])


    print("\nMetrics Summary:")

    print(df.drop(columns=["Top 3 Seeds"]))


    print("\nTop 3 Seeds (Experiments without 'baseline'):")
    no_baseline_df = df[~df["Experiment"].str.contains("baseline")]
    for _, row in no_baseline_df.iterrows():
        print(f"Experiment: {row['Experiment']}, Top 3 Seeds: {row['Top 3 Seeds']}")


    print("\nAll Top 3 Seeds (Unique across experiments):")
    print(top_seeds_set)

    return df


base_dir = "/home/users/ntu/tianle00/scratch/llmxfm-0.0/chemistry/logs/MOL/Lipophilicity_AstraZeneca"

top_k = 3

results_df = calculate_average_and_variance(base_dir, k=top_k)

