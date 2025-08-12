# Neural Networks and Deep Learning in Science: HW3 - Protein Family Classification

## Description
In this Homework, you will train a model (on protein sequences) to perform a multiclass classification task using the PFam seed dataset.

Part 1: Use the given pretrained model, i.e., Protein BERT, and fine-tune it ((https://huggingface.co/Rostlab/prot_bert)). This should be your baseline model.

Part 2: Try other models from Hugging Face and beat your baseline. Your Kaggle submission should come from your best-performing model.

Bonus Q: Use protein embeddings and visualize protein clusters with UMAP/tSNE.

You will be provided with the dataset and a skeleton code(s) that you must complete to conduct baseline training successfully, followed by fine-tuning.

Read more about the data here:

[1] Bileschi, M. L., Belanger, D., Bryant, D. H., Sanderson, T., Carter, B., Sculley, D., … & Colwell, L. J. (2022). Using deep learning to annotate the protein universe. Nature Biotechnology, 40(6), 932-937.

## Evaluation
The evaluation metric for this competition is accuracy score (sklearn.metrics.accuracy_score).

### My submission scored 0.99759 in accuracy
