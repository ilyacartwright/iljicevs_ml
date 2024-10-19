import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importances, feature_names, output_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names, y=importances)
    plt.xlabel("Признаки")
    plt.ylabel("Важность")
    plt.title("Важность признаков")
    plt.savefig(output_path)
    plt.close()
