# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

RandomForestClassifier model trained on the publicly available Census Bureau data [here](https://archive.ics.uci.edu/dataset/20/census+income).

- Model Type: RandomForestClassifier
- Trained with scikit-learn version: 1.3.2
- Model Hyperparameters: Default scikit-learn hyperparameters

## Intended Use

For classifying individuals into two salary categories: <=50K and >50K.

## Training Data

- Size: 26048
- Features: ``age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country``
- Target: salary

## Evaluation Data

- Size:  6513
- Features: same as training
- Target: salary

## Metrics
- Precision: 0.7418639053254438
- Recall: 0.6384468491406747
- Fbeta: 0.6862812179267875

## Ethical Considerations
- Potential bias in the dataset: The dataset may have potential bias as it might be imbalanced toward a specific salary category. Further investigation and collaboration with subject matter experts are recommended to address potential bias.
- Impact on individuals: Consider the impact of the model's predictions on individuals, especially if used in decision-making processes.

## Caveats and Recommendations
- The model's performance may vary on different subgroups of the population. Additional analysis on data slices is recommended to understand performance variations.
- Regularly update the model with new data to ensure its relevance and accuracy.