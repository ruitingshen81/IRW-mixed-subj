## Data Processing Notes

- First, all exact duplicate rows were removed from the dataset.

- Then, we checked whether any respondent answered the same item more than once.  
  For these cases, all corresponding rows for that person–item combination were removed entirely to avoid ambiguity.

- After processing, each respondent has at most one response per item, and the dataset contains only unique and valid observations.


### Output Columns (this may be included for general readme)

- `id`: person id
- `held_out_item`: The item being predicted
- `held_out_item_text`: Text of the held-out item
- `true_resp`: Ground truth response for the held-out item
- `generated_resp`: Model-predicted response
- `rationale`: Model-generated explanation for the prediction
- `n_observed_items`: Number of observed items used in the prompt
- `prompt`: Full prompt sent to the model

