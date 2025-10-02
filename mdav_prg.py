import os
import numpy as np
import pandas as pd
from collections import defaultdict

# === Constants ===
INPUT_FILE = "health_ai_mdav_demo.csv"
K = 4
QIS = ["Age", "ZIP", "SystolicBP", "BMI"]
IC = ["Sex"]
SCALING = "none"  # options: "minmax", "zscore", none
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
