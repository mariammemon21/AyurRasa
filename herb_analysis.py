import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
import streamlit as st

def run_herb_analysis():
    file_path = "D:\\IOT\\VADODARA\\AllRasaPS\\AllRasaPS\\dataPS.csv"
    df = pd.read_csv(file_path)
    st.write("Data Preview:")
    st.write(df.head())
    
    herb_fingerprints = np.array(df)
    st.write("Herb Molecular Fingerprint:")
    st.write(herb_fingerprints)
    
    def tanimoto_similarity(fingerprint1, fingerprint2):
        intersection = np.sum(np.minimum(fingerprint1, fingerprint2))
        union = np.sum(np.maximum(fingerprint1, fingerprint2))
        similarity = intersection / union
        return similarity
    
    taste_mapping = {
        (0.7, 1.0): 'Sweet',
        (0.5, 0.7): 'Sour',
        (0.3, 0.5): 'Bitter',
        (0.1, 0.3): 'Salty',
        (0.0, 0.1): 'Umami'
    }

    best_results = {}
    num_compounds = len(herb_fingerprints)

    for i in range(num_compounds):
        similarity_scores = []
        for j in range(num_compounds):
            if i != j:
                similarity_score = tanimoto_similarity(herb_fingerprints[i, :-3], herb_fingerprints[j, :-3])
                taste_name = 'Unknown taste'
                for range_values, name in taste_mapping.items():
                    if range_values[0] <= similarity_score <= range_values[1]:
                        taste_name = name
                        break
                similarity_scores.append((similarity_score, f"Compound {j + 1}: {taste_name}"))
        top_three = sorted(similarity_scores, reverse=True)[:3]
        best_results[f"Compound {i + 1}"] = top_three

    st.write("Best Three Results for Each Compound:")
    for compound, results in best_results.items():
        st.write(f"Results for {compound}:")
        for result in results:
            similarity_score, result_text = result
            st.write(f"{result_text} - Similarity Score: {similarity_score:.2f}")
        st.write("")

    st.write("Tanimoto Similarity Between Compounds:")
    for i in range(num_compounds):
        for j in range(i + 1, num_compounds):
            similarity_score = tanimoto_similarity(herb_fingerprints[i, :-3], herb_fingerprints[j, :-3])
            st.write(f"Tanimoto Similarity between Compound {i + 1} and Compound {j + 1}: {similarity_score}")
    
    pca = PCA(random_state=42)
    embedding_pca = pca.fit_transform(herb_fingerprints[:, :-3])
    st.write("PCA Embedding Shape:", embedding_pca.shape)
    
    plt.figure(figsize=(8, 6))
    if embedding_pca.shape[1] == 1:
        plt.scatter(embedding_pca[:, 0], np.zeros_like(embedding_pca[:, 0]))
    else:
        plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1])
    plt.title('PCA Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)

    st.write("Herb Fingerprints Shape:", herb_fingerprints.shape)

    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels = clustering.fit_predict(herb_fingerprints[:, :-3])
    distances = hierarchy.distance.pdist(herb_fingerprints[:, :-3])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Compounds')
    plt.ylabel('Distance')
    dendrogram = hierarchy.dendrogram(hierarchy.linkage(distances, method='ward'))

    plt.subplot(1, 2, 2)
    plt.title('t-SNE Visualization')
    tsne = TSNE(n_components=2, perplexity=1, random_state=42)
    embedding_tsne = tsne.fit_transform(herb_fingerprints[:, :-3])
    plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    st.pyplot(plt)
