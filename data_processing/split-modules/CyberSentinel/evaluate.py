import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.cm import Blues

def evaluate_model(model, test_data, test_labels, class_names=None, custom_metrics=None):
    # Evaluate the model
    scores = model.evaluate(test_data, test_labels, verbose=0)
    
    # Generate predictions
    y_pred = model.predict(test_data)
    y_pred = np.round(y_pred)
    
    # Confusion Matrix
    confusion_mat = confusion_matrix(test_labels, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)
    
    # Plot Confusion Matrix with labels
    if class_names:
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_mat, interpolation='nearest', cmap=Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(confusion_mat.shape[1]),
               yticks=np.arange(confusion_mat.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title="Confusion Matrix",
               ylabel='True label',
               xlabel='Predicted label')
        
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                ax.text(j, i, format(confusion_mat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if confusion_mat[i, j] > thresh else "black")
        fig.tight_layout()
        plt.show()
    
    # Classification Report
    cr = classification_report(test_labels, y_pred, target_names=class_names)
    print("Classification Report:")
    print(cr)
    
    # Custom Metrics
    if custom_metrics:
        custom_scores = {}
        for metric in custom_metrics:
            custom_scores[metric.__name__] = metric(test_labels, y_pred)
        return scores, custom_scores
    else:
        return scores

def model_interpretability_analysis(model, test_data, background_data=None):
    # Using SHAP for model interpretability
    if background_data is None:
        background_data = test_data[:100]
    
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data)
    
    # Plot SHAP values
    shap.summary_plot(shap_values, test_data, plot_type="bar")
