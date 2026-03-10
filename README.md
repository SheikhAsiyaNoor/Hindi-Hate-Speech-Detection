# Hindi Hate Speech Detection
This is a Python project for a Hindi hate speech detection model with performance similar to SOTA models.

# How to use
1. Run `pip install torch transformers pandas numpy scikit-learn`
2. Train the model - run (in main directory) `python step4_task2_multilabel.py`
3. Test it with inputs - run `python test_cli.py`

# References
* Model: [https://huggingface.co/google/muril-base-cased](google/muril-base-cased)
* Paper: [https://arxiv.org/pdf/2103.10730](*MuRIL: Multilingual Representations for Indian Languages*)
* Dataset: [https://www.kaggle.com/datasets/harithapliyal/hate-speech-in-hindi](*Kaggle hate speech in hindi dataset*)

# Estimated training time
*~30-40 minutes* for training
