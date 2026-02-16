from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file)
    df_numeric = df.select_dtypes(include=['int64', 'float64']).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)
    silhouette = silhouette_score(scaled_data, df["Cluster"])

    labels = {0: "Budget Shoppers", 1: "High Spenders", 2: "Occasional Buyers", 3: "Loyal Customers"}
    df["Cluster_Label"] = df["Cluster"].map(labels)
    df.to_csv(os.path.join(UPLOAD_FOLDER, "clustered_customers_labeled.csv"), index=False)
    df.to_excel(os.path.join(UPLOAD_FOLDER, "clustered_customers_labeled.xlsx"), index=False)

    sse = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(scaled_data)
        sse.append(km.inertia_)
    plt.figure()
    plt.plot(range(1, 11), sse, marker='o', linestyle='--')
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "elbow.png"))
    plt.close()

    plot_data = df_numeric.copy()
    plot_data["Cluster"] = df["Cluster"]
    num_cols = [c for c in df_numeric.columns if c != 'CustomerID'][:4]
    n = len(num_cols)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    for i, col1 in enumerate(num_cols):
        for j, col2 in enumerate(num_cols):
            ax = axes[i, j] if n > 1 else axes
            for cluster in sorted(plot_data["Cluster"].unique()):
                mask = plot_data["Cluster"] == cluster
                if i == j:
                    ax.hist(plot_data.loc[mask, col1], alpha=0.5, color=colors[cluster % len(colors)], label=labels[cluster])
                else:
                    ax.scatter(plot_data.loc[mask, col2], plot_data.loc[mask, col1], alpha=0.6, color=colors[cluster % len(colors)], label=labels[cluster], s=20)
            if i == n - 1:
                ax.set_xlabel(col2, fontsize=8)
            if j == 0:
                ax.set_ylabel(col1, fontsize=8)
            ax.tick_params(labelsize=6)
    handles, lbls = axes[0, 0].get_legend_handles_labels() if n > 1 else axes.get_legend_handles_labels()
    fig.legend(handles, lbls, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "pairplot.png"))
    plt.close('all')

    pca_2d = PCA(n_components=2).fit_transform(scaled_data)
    plt.figure()
    for label in sorted(set(df["Cluster"])):
        plt.scatter(pca_2d[df["Cluster"] == label, 0], pca_2d[df["Cluster"] == label, 1], label=labels[label])
    plt.legend()
    plt.title("PCA - 2D")
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "pca_2d.png"))
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D
    pca_3d = PCA(n_components=3).fit_transform(scaled_data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in sorted(set(df["Cluster"])):
        ax.scatter(
            pca_3d[df["Cluster"] == label, 0],
            pca_3d[df["Cluster"] == label, 1],
            pca_3d[df["Cluster"] == label, 2],
            label=labels[label]
        )
    ax.set_title("PCA - 3D")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "pca_3d.png"))
    plt.close()

    cluster_counts = df["Cluster_Label"].value_counts()
    plt.figure()
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
    plt.title("Customer Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "cluster_distribution_pie.png"))
    plt.close()

    plt.figure()
    plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.Set3.colors)
    plt.title("Customer Count per Cluster")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "cluster_distribution_bar.png"))
    plt.close()

    customers_data = df.to_dict(orient='records')
    columns = df.columns.tolist()

    return render_template(
        "result.html",
        elbow_img="static/uploads/elbow.png",
        cluster_img="static/uploads/pairplot.png",
        pca_2d_img="static/uploads/pca_2d.png",
        pca_3d_img="static/uploads/pca_3d.png",
        pie_chart_img="static/uploads/cluster_distribution_pie.png",
        bar_chart_img="static/uploads/cluster_distribution_bar.png",
        silhouette_score=round(silhouette, 3),
        customers_data=customers_data,
        columns=columns,
        total_customers=len(df)
    )

if __name__ == "__main__":
    app.run(debug=True)
