# Hindi Hate Speech Detection
This is a Python project for a Hindi hate speech detection model with performance similar to SOTA models.

# How to use
1. Run `pip install torch transformers pandas numpy scikit-learn`
2. Train the model - run (in main directory) `python step4_task2_multilabel.py`
3. Test it with inputs - run `python test_cli.py`

# References
* Model: [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased)
* Paper: [*MuRIL: Multilingual Representations for Indian Languages*](https://arxiv.org/pdf/2103.10730)
* Dataset: [*Kaggle hate speech in hindi dataset*](https://www.kaggle.com/datasets/harithapliyal/hate-speech-in-hindi)

# Estimated training time
*~30-40 minutes* for training
