## Data Processing Notes

- First, all exact duplicate rows were removed from the dataset.

- Then, we checked whether any respondent answered the same item more than once.  
  For these cases, all corresponding rows for that person–item combination were removed entirely to avoid ambiguity.

- After processing, each respondent has at most one response per item, and the dataset contains only unique and valid observations.

## Data Augmentation Notes

- The output data is generated using a leave-one-item-out procedure at the person level.
- For each person, one item response is held out at a time, and the remaining item responses are treated as observed data.
- A prompt is constructed using the observed item-response pairs, along with the held-out item text.
- The model is asked to predict the held-out response (0 or 1) based on the observed responses.
- Each row in the output corresponds to one prediction task (i.e., one held-out item for one person).

### Output Columns (this may be included for general readme)

- `id`: person id
- `held_out_item`: The item being predicted
- `held_out_item_text`: Text of the held-out item
- `true_resp`: Ground truth response for the held-out item
- `generated_resp`: Model-predicted response
- `rationale`: Model-generated explanation for the prediction
- `n_observed_items`: Number of observed items used in the prompt
- `prompt`: Full prompt sent to the model

