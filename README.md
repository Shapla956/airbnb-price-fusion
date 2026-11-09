# Fusing Photos, Text and Listing Details to Predict Airbnb Price Levels

This is the code for my individual project (Part 2) in the course Techniques for Multimodal Data Analysis. The goal is to predict whether a Melbourne Airbnb listing is budget, mid-range or premium, using three types of information about the listing: its main photo, its text and its listing details.

I first built the models planned in my proposal: one model for each modality on its own, and then early, late and attention-based fusion. These fusion models were only a little better than the tabular model, so I added some improvements from the course (CLIP image features, stacking, gated fusion and a fusion ensemble). The final model was chosen on the validation set.

| Modality | Where it comes from | How it is turned into numbers |
|---|---|---|
| Image | the main listing photo (`picture_url`) | EfficientNet-B0 (1,280 numbers, as planned) and CLIP ViT-B/32 (512 numbers, used in the improved and final models) |
| Text | the title and the description | BERT mean pooling (768 numbers), with BERT [CLS] for comparison |
| Tabular | room type, capacity, bedrooms, bathrooms, amenities, reviews and area | cleaned, scaled and one-hot encoded |

## What is in this repository

* `airbnb_price_fusion.ipynb`: the whole project (data, features, models and evaluation), saved with all outputs
* `requirements.txt`: the Python libraries needed
* `results/figures`: the figures from the notebook (PNG)
* `results/tables`: the tables from the notebook (CSV)
* `README.md`: this file

## Data

The data comes from Inside Airbnb, Melbourne snapshot of 16 September 2026 (https://insideairbnb.com/get-the-data/), under the Creative Commons Attribution 4.0 licence (CC BY 4.0).

You do not need to download anything yourself. The notebook reads the file straight from this link:
`https://data.insideairbnb.com/australia/vic/melbourne/2026-09-16/data/listings.csv.gz`

To protect privacy, all columns about the host and the exact latitude and longitude are removed at the start. Emails, phone numbers and links in the text are replaced with placeholder words. The photos are only used to make features and are never stored or published. No raw data is kept in this repository.

## How to reproduce the results

1. Open `airbnb_price_fusion.ipynb` in Google Colab and select a GPU: Runtime > Change runtime type > T4 GPU.
2. If you only want to check that the code works, set `"quick_run": 1500` in the settings cell (Section 1) and click Runtime > Run all. This uses a small part of the data and takes around 15 to 20 minutes.
3. To get the full results, change it back to `"quick_run": None` and click Run all again. This takes around 1.5 to 2.5 hours, mostly for downloading the photos, which only happens once.
4. When Colab asks for access to Google Drive, allow it. All features, tables and figures are saved in `MyDrive/airbnb_fusion/`.
5. If Colab disconnects, just run all cells again. The photo features are saved in small parts, so the run continues from where it stopped.

If you prefer to run it outside Colab, install the libraries with `pip install -r requirements.txt` and open the notebook in Jupyter. A GPU is strongly recommended. In that case the results are saved in `airbnb_fusion_outputs/`.

All settings are kept in one dictionary (`CFG`) and saved to `config.json`. The random seeds are fixed (42, 43 and 44), and every model is trained with all three. Two full runs gave the same numbers. Very small differences can still happen because of the GPU, or because some photos are no longer online.

## What the notebook does

| Section | Content |
|---|---|
| 2 to 3 | Load the data, remove personal data and make the three price levels (tertiles) |
| 4 to 5 | Data quality checks and preprocessing of each modality |
| 6 | Grouped 70/15/15 split, so listings that share a description or photo stay in the same set |
| 7 to 8 | Feature extraction, photo quality checks and a comparison of feature extractors |
| 9 | Single-modality models, early fusion and late fusion (XGBoost) and attention fusion (PyTorch), with settings chosen on validation using 3 seeds |
| 9.4 | Improvements after the first results: CLIP image features, stacking, gated fusion, auxiliary losses and a fusion ensemble; the final model is chosen on the validation set |
| 10 to 11 | Test metrics (accuracy, macro precision, recall and F1, ROC AUC, log loss, ECE, severe errors), seed comparison, confusion matrices, reliability diagram and overfitting checks |
| 12 | Tests with a missing, shuffled or noisy modality, attention weights and CKA |
| 13 to 14 | Bias and fairness by area and room type, results with and without area, and unseen areas |
| 15 to 16 | Failure cases and efficiency |

## Main results (test set, average over 3 seeds)

| Model | Macro F1 |
|---|---|
| Image only | 0.504 |
| Text only | 0.551 |
| Tabular only (best single modality) | 0.714 |
| Early / late / attention fusion (planned) | 0.715 / 0.716 / 0.712 |
| Stacking (CLIP) | 0.730 |
| Fusion ensemble (CLIP), final model | 0.730 |

The final model is the average of three models: early fusion (CLIP), stacking (CLIP) and tuned gated fusion (CLIP). It is 0.016 better than the best single modality, which is more than 2 times the variation between seeds, so the success criterion from my proposal is met. The notebook in this repository still has all the outputs from the full run.

## Main libraries

PyTorch and torchvision (EfficientNet-B0 and the fusion networks), Hugging Face Transformers (BERT and CLIP), XGBoost, scikit-learn, scikit-image, pandas and matplotlib.

## References

* Inside Airbnb (2026). Melbourne listings, 16 September 2026. https://insideairbnb.com/get-the-data/ (CC BY 4.0)
* Liang, P. P., Zadeh, A. and Morency, L. P. (2024). Foundations and trends in multimodal machine learning: Principles, challenges, and open questions. ACM Computing Surveys, 56(10).
* Szanto, A. and Mehta, N. (2018). A host of troubles: Re-identifying Airbnb hosts using public data. Technology Science. https://techscience.org/a/2018100902/
* Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). ICML.
* Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
* Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.
* Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
* Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12.
