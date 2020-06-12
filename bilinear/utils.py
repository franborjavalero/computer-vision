import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def export_metrics_values(file_name, acc_train, acc_test, loss_train, loss_test, decimals=5):
    metrics_values = np.round(
        np.array([
            acc_train,
            acc_test,
            loss_train,
            loss_test
        ]), 
        decimals=decimals,
    )
    
    metrics_labels = [
        "acc_train",
        "acc_test",
        "loss_train",
        "loss_test",
    ]
    
    df = pd.DataFrame(
        metrics_values.T,
        columns=metrics_labels,
    )

    df.to_csv(file_name, sep=',', encoding='utf-8', index=False, header=True)
