# MAIO_Assignment 2
Group Z: Dominic Behling, Filippo Besana, Dominik Eder, Chang Liu

## Task 1: Transfer Impact Assessment - Söderstad University Hospital AI Radiology System

### 1. Data Flow Sketch

### 2. Risk Analysis

### 3. Transfer Tool & Safeguards

### 4. Conclusion & Accountability

## Task 2: MDAV (Maximum Distance to Average Vector) Microaggregation Algorithm

### 1. Code
The following code implements the MDAV (Maximum Distance to Average Vector) Microaggregation Algorithm.
For the other tests performed, see GitHub repository: https://github.com/Domi-nik15/MAIO_Assignment2

**Source:** [`mdav_prg.py`](./mdav_prg.py)

```python
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# === Constants ===
INPUT_FILE = "health_ai_mdav_demo.csv"
K = 4
QIS = ["Age", "ZIP", "SystolicBP", "BMI"]
IC = ["Sex"]
SCALING = "zscore"  # options: "minmax", "zscore", none
OUTPUT_DIR = "mdav_output_k4_none"
TARGET_VAR = "Diagnosis"


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def scaled_array(df, cols, method='minmax'):
    arr = df[cols].astype(float).to_numpy(copy=True)
    if method is None:
        return arr
    if method == 'zscore':
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std[std == 0] = 1.0
        return (arr - mean) / std
    # default minmax
    minv = np.nanmin(arr, axis=0)
    maxv = np.nanmax(arr, axis=0)
    rng = maxv - minv
    rng[rng == 0] = 1.0
    return (arr - minv) / rng


def euclidean(a, b):
    return np.linalg.norm(a - b)

class MDAV:
    def __init__(self, df, qis, ic, k=3, scaling='minmax', output_dir='mdav_output'):
        self.df_original = df.copy().reset_index(drop=True)
        self.qis = qis
        self.ic = ic
        self.k = int(k)
        self.scaling = scaling
        self.output_dir = output_dir
        ensure_output_dir(self.output_dir)

        self.df_work = self.df_original.copy()
        self.scaled = scaled_array(self.df_work, self.qis, method=self.scaling)

        self.N = len(self.df_work)
        self.unassigned_idx = list(range(self.N))
        self.cluster_assignments = dict()
        self.cluster_members = defaultdict(list)
        self.current_cluster_id = 1
        self.iteration = 0

    def compute_centroid(self, indices):
        return np.nanmean(self.scaled[indices], axis=0)

    def farthest_point(self, indices, ref_point):
        best_idx = None
        best_dist = -1
        for i in indices:
            d = euclidean(self.scaled[i], ref_point)
            if d > best_dist:
                best_dist = d
                best_idx = i
        return best_idx
    
    def k_nearest_to_point(self, point_vector, candidates, k):
        distances = [(i, euclidean(self.scaled[i], point_vector)) for i in candidates]
        distances.sort(key=lambda x: x[1])
        selected = [i for i, _ in distances[:k]]
        return selected

    def k_nearest(self, center_idx, candidates, k):
        arr_c = self.scaled[center_idx]
        return self.k_nearest_to_point(arr_c, candidates, k)

    def form_cluster(self, members):
        cid = self.current_cluster_id
        for i in members:
            self.cluster_assignments[i] = cid
            self.cluster_members[cid].append(i)
            if i in self.unassigned_idx:
                self.unassigned_idx.remove(i)
        self.current_cluster_id += 1

    def output_iteration_csv(self):
        df_out = self.df_work.copy()
        for cid, members in self.cluster_members.items():
            if not members:
                continue
            mean_vals = self.df_work.loc[members, self.qis].astype(float).mean(axis=0).to_dict()
            for i in members:
                for col in self.qis:
                    df_out.at[i, col] = mean_vals[col]
        filename = os.path.join(self.output_dir, f'iteration_{self.iteration:03d}_anonymized.csv')
        df_out.to_csv(filename, index=False)
        print(f"Wrote iteration {self.iteration} anonymized CSV: {filename}")

    def run(self):
        if self.N < self.k:
            raise ValueError(f"N={self.N} < k={self.k}. Cannot form any cluster.")
        
        for col in self.ic:
            self.df_work[col] = "*"

        while len(self.unassigned_idx) >= 3 * self.k:
            centroid = self.compute_centroid(self.unassigned_idx)
            a = self.farthest_point(self.unassigned_idx, centroid)
            b = self.farthest_point(self.unassigned_idx, self.scaled[a])

            group_a = self.k_nearest(a, self.unassigned_idx, self.k)
            self.form_cluster(group_a)

            group_b = self.k_nearest(b, self.unassigned_idx, self.k)
            self.form_cluster(group_b)

            self.output_iteration_csv()
            self.iteration += 1

        rem = len(self.unassigned_idx)
        if rem > 0:
            if rem >= self.k:
                while len(self.unassigned_idx) >= self.k:
                    centroid = self.compute_centroid(self.unassigned_idx)
                    a = self.farthest_point(self.unassigned_idx, centroid)
                    group = self.k_nearest(a, self.unassigned_idx, self.k)
                    self.form_cluster(group)
            leftovers = list(self.unassigned_idx)
            if leftovers:
                cluster_centroids = {}
                for cid, members in self.cluster_members.items():
                    cluster_centroids[cid] = np.nanmean(self.scaled[members], axis=0)
                for i in leftovers:
                    best_cid = None
                    best_dist = float('inf')
                    for cid, cc in cluster_centroids.items():
                        d = euclidean(self.scaled[i], cc)
                        if d < best_dist:
                            best_dist = d
                            best_cid = cid
                    self.cluster_assignments[i] = best_cid
                    self.cluster_members[best_cid].append(i)
                    self.unassigned_idx.remove(i)
                self.iteration += 1
                self.output_iteration_csv()

        self.output_final_files()

    def output_final_files(self):
        # Create final anonymized CSV
        df_final = self.df_work.copy()
        cluster_means = {}
        for cid, members in self.cluster_members.items():
            means = self.df_work.loc[members, self.qis].astype(float).mean(axis=0)
            cluster_means[cid] = means
            for i in members:
                for col in self.qis:
                    df_final.at[i, col] = means[col]

        final_csv = os.path.join(self.output_dir, 'final_anonymized.csv')
        df_final.to_csv(final_csv, index=False)
        print(f"Wrote final anonymized CSV: {final_csv}")

        # Create cluster assignment file
        rows = []
        for i in range(self.N):
            recid = self.df_work.at[i, 'ID']
            cid = self.cluster_assignments.get(i, -1)
            rows.append({'RecordID': recid, 'ClusterID': cid})
        df_assign = pd.DataFrame(rows)
        assign_file = os.path.join(self.output_dir, 'cluster_assignments.csv')
        df_assign.to_csv(assign_file, index=False)
        print(f"Wrote cluster assignment CSV: {assign_file}")

        # Create cluster summary file
        summary_rows = []
        for cid, members in self.cluster_members.items():
            row = {"ClusterID": cid}
            means = self.df_work.loc[members, self.qis].astype(float).mean(axis=0)
            for col in self.qis:
                row[col] = means[col]
                if TARGET_VAR in self.df_work.columns:
                    unique_vals = self.df_work.loc[members, TARGET_VAR].dropna().unique()
                    row[TARGET_VAR] = ",".join(sorted(map(str, unique_vals)))
                summary_rows.append(row)
        df_summary = pd.DataFrame(summary_rows).drop_duplicates()
        summary_file = os.path.join(self.output_dir, 'cluster_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        print(f"Wrote cluster summary CSV: {summary_file}")

        # Create report
        sse = 0.0
        cluster_radii = []
        for cid, members in self.cluster_members.items():
            members = list(members)
            centroid = np.nanmean(self.df_work.loc[members, self.qis].astype(float).to_numpy(), axis=0)
            arr = self.df_work.loc[members, self.qis].astype(float).to_numpy()
            diffs = arr - centroid
            sse += np.sum(np.square(diffs))
            dists = np.linalg.norm(diffs, axis=1)
            if len(dists) > 0:
                cluster_radii.append(np.max(dists))

        avg_radius = float(np.mean(cluster_radii)) if cluster_radii else 0.0

        sizes = {cid: len(members) for cid, members in self.cluster_members.items()}
        privacy_ok = all(sz >= self.k for sz in sizes.values())

        report_lines = []
        report_lines.append(f"MDAV microaggregation report")
        report_lines.append(f"Input file rows: {self.N}")
        report_lines.append(f"Quasi-identifiers used (distance computation): {self.qis}")
        report_lines.append(f"k (anonymity parameter): {self.k}")
        report_lines.append("")
        report_lines.append("Method notes:")
        report_lines.append(" - The values of the insensitive columns were suppressed to '*'.")
        report_lines.append(" - Scaling used for distance computations: {}".format(self.scaling))
        report_lines.append(" - MDAV procedure: while |R| >= 3k: pick centroid of R, find farthest A, farthest B from A, form k-nearest groups around A and B. Then handle remainder.")
        report_lines.append("")
        report_lines.append("Privacy check:")
        report_lines.append(f" - number of clusters formed: {len(self.cluster_members)}")
        report_lines.append(f" - cluster sizes (ClusterID: size): {sizes}")
        report_lines.append(f" - privacy satisfied (every cluster size >= k): {privacy_ok}")
        report_lines.append("")
        report_lines.append("Quality metrics:")
        report_lines.append(f" - SSE (sum of squared errors) in original QI space: {sse:.6f}")
        report_lines.append(f" - average cluster radius (mean of per-cluster avg distances): {avg_radius:.6f}")

        report_text = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, 'mdav_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"Wrote report: {report_file}")
        print(report_text)


def main():
    df = pd.read_csv(INPUT_FILE)
    for col in QIS:
        if col not in df.columns:
            raise KeyError(f"QI column '{col}' not found in input CSV columns: {list(df.columns)}")
    m = MDAV(df, qis=QIS, ic=IC, k=K, scaling=(None if SCALING=='none' else SCALING), output_dir=OUTPUT_DIR)
    m.run()


if __name__ == '__main__':
    main()

```

### 2. Anonymized CSV
#### mdav_output_k4_zscore
Iteration 1:
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 27    | *     | 53116   |        108   | 27.1   | Hypertension |
|    2 | 66    | *     | 53118   |        123   | 25.2   | Migraine     |
|    3 | 59    | *     | 53118   |        126   | 22.8   | Asthma       |
|    4 | 47    | *     | 53116   |        108   | 24.9   | Diabetes     |
|    5 | 46    | *     | 53118   |        110   | 25     | Migraine     |
|    6 | 64.75 | *     | 53117.8 |        121.5 | 28.675 | Cancer       |
|    7 | 26    | *     | 53116   |         90   | 25.1   | Diabetes     |
|    8 | 61    | *     | 53116   |        127   | 26.4   | Diabetes     |
|    9 | 33    | *     | 53117   |        101   | 27.2   | Cancer       |
|   10 | 28.75 | *     | 53116   |         93   | 21.625 | Flu          |
|   11 | 52    | *     | 53115   |        122   | 23.2   | Hypertension |
|   12 | 77    | *     | 53115   |        137   | 24.3   | Diabetes     |
|   13 | 63    | *     | 53118   |        130   | 24.6   | Cancer       |
|   14 | 65    | *     | 53117   |        124   | 22.3   | Migraine     |
|   15 | 64.75 | *     | 53117.8 |        121.5 | 28.675 | Asthma       |
|   16 | 66    | *     | 53118   |        125   | 25.3   | Diabetes     |
|   17 | 51    | *     | 53117   |        100   | 21.1   | Cancer       |
|   18 | 28.75 | *     | 53116   |         93   | 21.625 | Hypertension |
|   19 | 69    | *     | 53115   |        114   | 26.8   | Hypertension |
|   20 | 47    | *     | 53115   |        104   | 26.6   | Hypertension |
|   21 | 50    | *     | 53117   |        120   | 29.7   | Cancer       |
|   22 | 43    | *     | 53117   |        103   | 31.5   | Diabetes     |
|   23 | 32    | *     | 53117   |        101   | 24.5   | Flu          |
|   24 | 74    | *     | 53118   |        143   | 23.9   | Hypertension |
|   25 | 66    | *     | 53117   |        122   | 20.5   | Hypertension |
|   26 | 58    | *     | 53117   |        128   | 25.9   | Diabetes     |
|   27 | 44    | *     | 53117   |        103   | 22.2   | Asthma       |
|   28 | 68    | *     | 53116   |        125   | 24.9   | Hypertension |
|   29 | 53    | *     | 53115   |        108   | 23.4   | Diabetes     |
|   30 | 47    | *     | 53116   |        111   | 24.1   | Cancer       |
|   31 | 47    | *     | 53115   |        123   | 27.2   | Flu          |
|   32 | 28.75 | *     | 53116   |         93   | 21.625 | Asthma       |
|   33 | 27    | *     | 53119   |        107   | 22.7   | Diabetes     |
|   34 | 53    | *     | 53115   |        120   | 22.3   | Migraine     |
|   35 | 72    | *     | 53115   |        123   | 22.1   | Migraine     |
|   36 | 28.75 | *     | 53116   |         93   | 21.625 | Migraine     |
|   37 | 70    | *     | 53116   |        129   | 26     | Cancer       |
|   38 | 64.75 | *     | 53117.8 |        121.5 | 28.675 | Migraine     |
|   39 | 37    | *     | 53117   |        111   | 24.1   | Flu          |
|   40 | 64.75 | *     | 53117.8 |        121.5 | 28.675 | Hypertension |

Iteration 2:
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 27    | *     | 53116   |       108    | 27.1   | Hypertension |
|    2 | 66    | *     | 53118   |       123    | 25.2   | Migraine     |
|    3 | 59    | *     | 53118   |       126    | 22.8   | Asthma       |
|    4 | 47    | *     | 53116   |       108    | 24.9   | Diabetes     |
|    5 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Migraine     |
|    6 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Cancer       |
|    7 | 26    | *     | 53116   |        90    | 25.1   | Diabetes     |
|    8 | 61    | *     | 53116   |       127    | 26.4   | Diabetes     |
|    9 | 33    | *     | 53117   |       101    | 27.2   | Cancer       |
|   10 | 28.75 | *     | 53116   |        93    | 21.625 | Flu          |
|   11 | 52    | *     | 53115   |       122    | 23.2   | Hypertension |
|   12 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Diabetes     |
|   13 | 63    | *     | 53118   |       130    | 24.6   | Cancer       |
|   14 | 65    | *     | 53117   |       124    | 22.3   | Migraine     |
|   15 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Asthma       |
|   16 | 66    | *     | 53118   |       125    | 25.3   | Diabetes     |
|   17 | 51    | *     | 53117   |       100    | 21.1   | Cancer       |
|   18 | 28.75 | *     | 53116   |        93    | 21.625 | Hypertension |
|   19 | 69    | *     | 53115   |       114    | 26.8   | Hypertension |
|   20 | 47    | *     | 53115   |       104    | 26.6   | Hypertension |
|   21 | 50    | *     | 53117   |       120    | 29.7   | Cancer       |
|   22 | 43    | *     | 53117   |       103    | 31.5   | Diabetes     |
|   23 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   24 | 74    | *     | 53118   |       143    | 23.9   | Hypertension |
|   25 | 66    | *     | 53117   |       122    | 20.5   | Hypertension |
|   26 | 58    | *     | 53117   |       128    | 25.9   | Diabetes     |
|   27 | 44    | *     | 53117   |       103    | 22.2   | Asthma       |
|   28 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Hypertension |
|   29 | 53    | *     | 53115   |       108    | 23.4   | Diabetes     |
|   30 | 47    | *     | 53116   |       111    | 24.1   | Cancer       |
|   31 | 47    | *     | 53115   |       123    | 27.2   | Flu          |
|   32 | 28.75 | *     | 53116   |        93    | 21.625 | Asthma       |
|   33 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Diabetes     |
|   34 | 53    | *     | 53115   |       120    | 22.3   | Migraine     |
|   35 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Migraine     |
|   36 | 28.75 | *     | 53116   |        93    | 21.625 | Migraine     |
|   37 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Cancer       |
|   38 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Migraine     |
|   39 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   40 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Hypertension |

Iteration 3:
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 27    | *     | 53116   |       108    | 27.1   | Hypertension |
|    2 | 66    | *     | 53118   |       123    | 25.2   | Migraine     |
|    3 | 59    | *     | 53118   |       126    | 22.8   | Asthma       |
|    4 | 47    | *     | 53116   |       108    | 24.9   | Diabetes     |
|    5 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Migraine     |
|    6 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Cancer       |
|    7 | 26    | *     | 53116   |        90    | 25.1   | Diabetes     |
|    8 | 61    | *     | 53116   |       127    | 26.4   | Diabetes     |
|    9 | 33    | *     | 53117   |       101    | 27.2   | Cancer       |
|   10 | 28.75 | *     | 53116   |        93    | 21.625 | Flu          |
|   11 | 52    | *     | 53115   |       122    | 23.2   | Hypertension |
|   12 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Diabetes     |
|   13 | 63    | *     | 53118   |       130    | 24.6   | Cancer       |
|   14 | 65    | *     | 53117   |       124    | 22.3   | Migraine     |
|   15 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Asthma       |
|   16 | 66    | *     | 53118   |       125    | 25.3   | Diabetes     |
|   17 | 51    | *     | 53117   |       100    | 21.1   | Cancer       |
|   18 | 28.75 | *     | 53116   |        93    | 21.625 | Hypertension |
|   19 | 69    | *     | 53115   |       114    | 26.8   | Hypertension |
|   20 | 47    | *     | 53115   |       104    | 26.6   | Hypertension |
|   21 | 50    | *     | 53117   |       120    | 29.7   | Cancer       |
|   22 | 43    | *     | 53117   |       103    | 31.5   | Diabetes     |
|   23 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   24 | 74    | *     | 53118   |       143    | 23.9   | Hypertension |
|   25 | 66    | *     | 53117   |       122    | 20.5   | Hypertension |
|   26 | 58    | *     | 53117   |       128    | 25.9   | Diabetes     |
|   27 | 44    | *     | 53117   |       103    | 22.2   | Asthma       |
|   28 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Hypertension |
|   29 | 53    | *     | 53115   |       108    | 23.4   | Diabetes     |
|   30 | 47    | *     | 53116   |       111    | 24.1   | Cancer       |
|   31 | 47    | *     | 53115   |       123    | 27.2   | Flu          |
|   32 | 28.75 | *     | 53116   |        93    | 21.625 | Asthma       |
|   33 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Diabetes     |
|   34 | 53    | *     | 53115   |       120    | 22.3   | Migraine     |
|   35 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Migraine     |
|   36 | 28.75 | *     | 53116   |        93    | 21.625 | Migraine     |
|   37 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Cancer       |
|   38 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Migraine     |
|   39 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   40 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Hypertension |
PS P:\Master\Sem3\MAIO\Assigment\MAIO_Assignment2> & C:/Python313/python.exe p:/Master/Sem3/MAIO/Assigment/MAIO_Assignment2/test.py
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Hypertension |
|    2 | 66    | *     | 53118   |       123    | 25.2   | Migraine     |
|    3 | 65.5  | *     | 53118   |       131    | 24.15  | Asthma       |
|    4 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    5 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Migraine     |
|    6 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Cancer       |
|    7 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    8 | 61    | *     | 53116   |       127    | 26.4   | Diabetes     |
|    9 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Cancer       |
|   10 | 28.75 | *     | 53116   |        93    | 21.625 | Flu          |
|   11 | 52    | *     | 53115   |       122    | 23.2   | Hypertension |
|   12 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Diabetes     |
|   13 | 65.5  | *     | 53118   |       131    | 24.15  | Cancer       |
|   14 | 65    | *     | 53117   |       124    | 22.3   | Migraine     |
|   15 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Asthma       |
|   16 | 65.5  | *     | 53118   |       131    | 24.15  | Diabetes     |
|   17 | 51    | *     | 53117   |       100    | 21.1   | Cancer       |
|   18 | 28.75 | *     | 53116   |        93    | 21.625 | Hypertension |
|   19 | 69    | *     | 53115   |       114    | 26.8   | Hypertension |
|   20 | 47    | *     | 53115   |       104    | 26.6   | Hypertension |
|   21 | 50    | *     | 53117   |       120    | 29.7   | Cancer       |
|   22 | 43    | *     | 53117   |       103    | 31.5   | Diabetes     |
|   23 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   24 | 65.5  | *     | 53118   |       131    | 24.15  | Hypertension |
|   25 | 66    | *     | 53117   |       122    | 20.5   | Hypertension |
|   26 | 58    | *     | 53117   |       128    | 25.9   | Diabetes     |
|   27 | 44    | *     | 53117   |       103    | 22.2   | Asthma       |
|   28 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Hypertension |
|   29 | 53    | *     | 53115   |       108    | 23.4   | Diabetes     |
|   30 | 47    | *     | 53116   |       111    | 24.1   | Cancer       |
|   31 | 47    | *     | 53115   |       123    | 27.2   | Flu          |
|   32 | 28.75 | *     | 53116   |        93    | 21.625 | Asthma       |
|   33 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Diabetes     |
|   34 | 53    | *     | 53115   |       120    | 22.3   | Migraine     |
|   35 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Migraine     |
|   36 | 28.75 | *     | 53116   |        93    | 21.625 | Migraine     |
|   37 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Cancer       |
|   38 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Migraine     |
|   39 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   40 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Hypertension |

Iteration 4:
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Hypertension |
|    2 | 62    | *     | 53117.2 |       117.25 | 22.275 | Migraine     |
|    3 | 65.5  | *     | 53118   |       131    | 24.15  | Asthma       |
|    4 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    5 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Migraine     |
|    6 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Cancer       |
|    7 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    8 | 61    | *     | 53116   |       127    | 26.4   | Diabetes     |
|    9 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Cancer       |
|   10 | 28.75 | *     | 53116   |        93    | 21.625 | Flu          |
|   11 | 52    | *     | 53115   |       122    | 23.2   | Hypertension |
|   12 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Diabetes     |
|   13 | 65.5  | *     | 53118   |       131    | 24.15  | Cancer       |
|   14 | 62    | *     | 53117.2 |       117.25 | 22.275 | Migraine     |
|   15 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Asthma       |
|   16 | 65.5  | *     | 53118   |       131    | 24.15  | Diabetes     |
|   17 | 62    | *     | 53117.2 |       117.25 | 22.275 | Cancer       |
|   18 | 28.75 | *     | 53116   |        93    | 21.625 | Hypertension |
|   19 | 69    | *     | 53115   |       114    | 26.8   | Hypertension |
|   20 | 46.75 | *     | 53116   |       112.5  | 28.75  | Hypertension |
|   21 | 46.75 | *     | 53116   |       112.5  | 28.75  | Cancer       |
|   22 | 46.75 | *     | 53116   |       112.5  | 28.75  | Diabetes     |
|   23 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   24 | 65.5  | *     | 53118   |       131    | 24.15  | Hypertension |
|   25 | 62    | *     | 53117.2 |       117.25 | 22.275 | Hypertension |
|   26 | 58    | *     | 53117   |       128    | 25.9   | Diabetes     |
|   27 | 44    | *     | 53117   |       103    | 22.2   | Asthma       |
|   28 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Hypertension |
|   29 | 53    | *     | 53115   |       108    | 23.4   | Diabetes     |
|   30 | 47    | *     | 53116   |       111    | 24.1   | Cancer       |
|   31 | 46.75 | *     | 53116   |       112.5  | 28.75  | Flu          |
|   32 | 28.75 | *     | 53116   |        93    | 21.625 | Asthma       |
|   33 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Diabetes     |
|   34 | 53    | *     | 53115   |       120    | 22.3   | Migraine     |
|   35 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Migraine     |
|   36 | 28.75 | *     | 53116   |        93    | 21.625 | Migraine     |
|   37 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Cancer       |
|   38 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Migraine     |
|   39 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   40 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Hypertension |

Final File:
|   ID |   Age | Sex   |     ZIP |   SystolicBP |    BMI | Diagnosis    |
|-----:|------:|:------|--------:|-------------:|-------:|:-------------|
|    1 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Hypertension |
|    2 | 62    | *     | 53117.2 |       117.25 | 22.275 | Migraine     |
|    3 | 65.5  | *     | 53118   |       131    | 24.15  | Asthma       |
|    4 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    5 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Migraine     |
|    6 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Cancer       |
|    7 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Diabetes     |
|    8 | 60    | *     | 53115.8 |       122.75 | 25.575 | Diabetes     |
|    9 | 33.25 | *     | 53116.2 |       101.75 | 26.075 | Cancer       |
|   10 | 28.75 | *     | 53116   |        93    | 21.625 | Flu          |
|   11 | 60    | *     | 53115.8 |       122.75 | 25.575 | Hypertension |
|   12 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Diabetes     |
|   13 | 65.5  | *     | 53118   |       131    | 24.15  | Cancer       |
|   14 | 62    | *     | 53117.2 |       117.25 | 22.275 | Migraine     |
|   15 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Asthma       |
|   16 | 65.5  | *     | 53118   |       131    | 24.15  | Diabetes     |
|   17 | 62    | *     | 53117.2 |       117.25 | 22.275 | Cancer       |
|   18 | 28.75 | *     | 53116   |        93    | 21.625 | Hypertension |
|   19 | 60    | *     | 53115.8 |       122.75 | 25.575 | Hypertension |
|   20 | 46.75 | *     | 53116   |       112.5  | 28.75  | Hypertension |
|   21 | 46.75 | *     | 53116   |       112.5  | 28.75  | Cancer       |
|   22 | 46.75 | *     | 53116   |       112.5  | 28.75  | Diabetes     |
|   23 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   24 | 65.5  | *     | 53118   |       131    | 24.15  | Hypertension |
|   25 | 62    | *     | 53117.2 |       117.25 | 22.275 | Hypertension |
|   26 | 60    | *     | 53115.8 |       122.75 | 25.575 | Diabetes     |
|   27 | 49.25 | *     | 53115.8 |       110.5  | 23     | Asthma       |
|   28 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Hypertension |
|   29 | 49.25 | *     | 53115.8 |       110.5  | 23     | Diabetes     |
|   30 | 49.25 | *     | 53115.8 |       110.5  | 23     | Cancer       |
|   31 | 46.75 | *     | 53116   |       112.5  | 28.75  | Flu          |
|   32 | 28.75 | *     | 53116   |        93    | 21.625 | Asthma       |
|   33 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Diabetes     |
|   34 | 49.25 | *     | 53115.8 |       110.5  | 23     | Migraine     |
|   35 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Migraine     |
|   36 | 28.75 | *     | 53116   |        93    | 21.625 | Migraine     |
|   37 | 71.75 | *     | 53115.5 |       128.5  | 24.325 | Cancer       |
|   38 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Migraine     |
|   39 | 35.5  | *     | 53117.8 |       107.25 | 24.075 | Flu          |
|   40 | 64.75 | *     | 53117.8 |       121.5  | 28.675 | Hypertension |

Cluster Summary:
|   ClusterID |   Age | Diagnosis                             |     ZIP |   SystolicBP |    BMI |
|------------:|------:|:--------------------------------------|--------:|-------------:|-------:|    
|           1 | 28.75 | Asthma,Flu,Hypertension,Migraine      | 53116   |        93    | 21.625 |    
|           2 | 64.75 | Asthma,Cancer,Hypertension,Migraine   | 53117.8 |       121.5  | 28.675 |    
|           3 | 35.5  | Diabetes,Flu,Migraine                 | 53117.8 |       107.25 | 24.075 |    
|           4 | 71.75 | Cancer,Diabetes,Hypertension,Migraine | 53115.5 |       128.5  | 24.325 |    
|           5 | 65.5  | Asthma,Cancer,Diabetes,Hypertension   | 53118   |       131    | 24.15  |    
|           6 | 33.25 | Cancer,Diabetes,Hypertension          | 53116.2 |       101.75 | 26.075 |    
|           7 | 46.75 | Cancer,Diabetes,Flu,Hypertension      | 53116   |       112.5  | 28.75  |    
|           8 | 62    | Cancer,Hypertension,Migraine          | 53117.2 |       117.25 | 22.275 |    
|           9 | 49.25 | Asthma,Cancer,Diabetes,Migraine       | 53115.8 |       110.5  | 23     |    
|          10 | 60    | Diabetes,Hypertension                 | 53115.8 |       122.75 | 25.575 | 

### 3. Cluster Assignment File

|   RecordID |   ClusterID |
|-----------:|------------:|
|          1 |           6 |
|          2 |           8 |
|          3 |           5 |
|          4 |           6 |
|          5 |           3 |
|          6 |           2 |
|          7 |           6 |
|          8 |          10 |
|          9 |           6 |
|         10 |           1 |
|         11 |          10 |
|         12 |           4 |
|         13 |           5 |
|         14 |           8 |
|         15 |           2 |
|         16 |           5 |
|         17 |           8 |
|         18 |           1 |
|         19 |          10 |
|         20 |           7 |
|         21 |           7 |
|         22 |           7 |
|         23 |           3 |
|         24 |           5 |
|         25 |           8 |
|         26 |          10 |
|         27 |           9 |
|         28 |           4 |
|         29 |           9 |
|         30 |           9 |
|         31 |           7 |
|         32 |           1 |
|         33 |           3 |
|         34 |           9 |
|         35 |           4 |
|         36 |           1 |
|         37 |           4 |
|         38 |           2 |
|         39 |           3 |
|         40 |           2 |

### 4. Report

MDAV microaggregation report
Input file rows: 40
Quasi-identifiers used (distance computation): ['Age', 'ZIP', 'SystolicBP', 'BMI']
k (anonymity parameter): 4

Method notes:
 - The values of the insensitive columns were suppressed to '*'.
 - Scaling used for distance computations: zscore
 - MDAV procedure: while |R| >= 3k: pick centroid of R, find farthest A, farthest B from A, form k-nearest groups around A and B. Then handle remainder.

Privacy check:
 - number of clusters formed: 10
 - cluster sizes (ClusterID: size): {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4}
 - privacy satisfied (every cluster size >= k): True

Quality metrics:
 - SSE (sum of squared errors) in original QI space: 2998.992500
 - average cluster radius (mean of per-cluster avg distances): 12.100648