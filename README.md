# MSc-2 Thesis

## Machine Learning Based Analysis of Gene Expression and Lipid Composition of the Schizophrenia Brain

## Анализ при помощи машинного обучения экспрессии генов и содержания липидов в мозге при шизофрении

The molecular mechanisms of schizophrenia are still poorly understood. In this research, we apply machine learning techniques with the purpose of discovering what genes, lipids, and brain regions are relevant to this disease. We analyzed the dataset of gene expression and lipid composition post-mortem measurements in multiple brain regions in four healthy individuals and four individuals diagnosed with schizophrenia. We visualized the dataset using PCA (Principal Component Analysis). We applied random forest, logistic regression, and SVM (Support Vector Machine) machine learning classification algorithms to the dataset, treating the disease as a label. We used two approaches: the brain as an object and the region as an object.

The number of people is our dataset is small --- this is the limitation of such datasets, because it is hard to obtain biological samples, and at the same time is is easy to make many measurements from one sample. We address this problem by data augmentation. We estimate the distributions of molecule measurements and generate the synthetic dataset. Also, we find and incorporate an external dataset similar to ours.

We were able to estimate the importances of genes, lipids, and regions. Namely, we obtained the Gini importance according to the random forest algorithm and the permutational importances for logistic regression and SVM. The importances of the brain regions were computed as the accuracy of diagnosis prediction for such a region or as the sum of the importances of the molecules in the region. We found out that some molecules and regions distinguish the healthy and schizophrenic brains more than others. Since the molecular changes in schizophrenic brains are relatively subtle, the results obtained from different algorithms are not exactly the same.

See the full text and presentation for more details.

