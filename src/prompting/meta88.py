from pickle import FALSE
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pandas as pd
from chatlas import ChatOpenAI, parallel_chat_structured
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


DATA_PATH = Path("data/processed/meta88-person-specific-itemresp.csv")
OUT_DIR = Path("data/llm-out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_PATH = OUT_DIR / "meta88_llm.csv"
ERRORS_PATH = OUT_DIR / "meta88_llm_error.csv"


MODEL_NAME = "gpt-5.4-nano"
SAMPLE_N_PEOPLE = 2
RANDOM_SEED = 123

SYSTEM_PROMPT = """
    You are an expert in generating questionnaire responses based on the responses for other questions.

    Task:
    Given one person's observed responses to other items, predict that person's response
    to one held-out item.

    Response scale:
    - 0 = No
    - 1 = Yes

    Rules:
    - Use only the observed responses that are provided.
    - Return exactly one value: 0 or 1.
""".strip()


# structured output


class PredictedResponse(BaseModel):
    predicted_response: Literal[0, 1] = Field(
        description="Predicted binary response for the held-out item: 0 for No, 1 for Yes."
    )


# now load data functions


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def sample_people(df: pd.DataFrame, n_people: int, seed: int) -> pd.DataFrame:
    unique_ids = df["id"].drop_duplicates()
    sampled_ids = unique_ids.sample(n=n_people, random_state=seed)
    return df[df["id"].isin(sampled_ids)].copy()


# actual useer prompt
def build_user_prompt(person_df: pd.DataFrame, held_out_idx: int) -> tuple[str, dict]:
    # holding this one out for prediction
    held_out = person_df.iloc[held_out_idx]
    # those are the ones we condition on
    observed = person_df.drop(person_df.index[held_out_idx]).reset_index(drop=True)

    ###this will create a summary for each item that it's conditioned on for prediction
    observed_lines = []
    for _, row in observed.iterrows():
        observed_lines.append(
            f"Item ID: {row['item']}\n"
            f'Item text: "{row["item_text"]}"\n'
            f"Response: {row['resp']}"
        )
    # now putting those conditioned items together and put together the prompt for each held out
    observed_block = "\n\n".join(observed_lines)

    prompt = f"""
    Below are the observed responses for one person.
    Observed item responses:
    {observed_block}

    Now predict this person's response to the held-out item below.

    Held-out item ID: {held_out["item"]}
    Held-out item text: "{held_out["item_text"]}"

    Return the predicted response as 0 or 1.

""".strip()

    meta = {
        "id": held_out["id"],
        "held_out_item": held_out["item"],
        "held_out_item_text": held_out["item_text"],
        "true_resp": int(held_out["resp"]),
        "n_observed_items": int(len(observed)),
        "prompt": prompt,
    }

    return prompt, meta


def build_tasks(df: pd.DataFrame) -> list[dict]:
    tasks = []
    for _, person_df in df.groupby("id", sort=False):
        person_df = person_df.sort_values("item").reset_index(drop=True)
        # have to make sure that one person has more than 2 items answered
        if len(person_df) < 2:
            continue

        for i in range(len(person_df)):
            _, meta = build_user_prompt(person_df, i)
            tasks.append(meta)

    return tasks


# generate responses
async def generate_leave_one_out(
    tasks: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    chat = ChatOpenAI(
        model=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
    )
    generated_rows = []
    error_rows = []

    prompts = [task["prompt"] for task in tasks]

    results = await parallel_chat_structured(
        chat,
        prompts,
        PredictedResponse,
        max_active=10,
        rpm=300,
        on_error="continue",
    )

    for task, result in zip(tasks, results):
        if result is not None and hasattr(result, "data"):
            generated_rows.append(
                {
                    "id": task["id"],
                    "held_out_item": task["held_out_item"],
                    "held_out_item_text": task["held_out_item_text"],
                    "true_resp": task["true_resp"],
                    "generated_resp": int(result.data.predicted_response),
                    "n_observed_items": task["n_observed_items"],
                    "prompt": task["prompt"],
                }
            )
        else:
            error_rows.append(
                {
                    "id": task["id"],
                    "held_out_item": task["held_out_item"],
                    "held_out_item_text": task["held_out_item_text"],
                    "true_resp": task["true_resp"],
                    "n_observed_items": task["n_observed_items"],
                    "prompt": task["prompt"],
                    "error": repr(result),
                }
            )

    return pd.DataFrame(generated_rows), pd.DataFrame(error_rows)


###MAIN####
async def main() -> None:
    # change it to False when generating the actual the full dataset
    USE_SAMPLED_DATA = False
    df = load_data(DATA_PATH)

    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Unique persons before sampling: {df['id'].nunique()}")

    if USE_SAMPLED_DATA:
        df = sample_people(df, n_people=SAMPLE_N_PEOPLE, seed=RANDOM_SEED)
        print(f"Using sampled data with {SAMPLE_N_PEOPLE} people")
    else:
        print("Using full dataset")

    print(f"Unique persons used: {df['id'].nunique()}")
    print(f"Rows used: {len(df)}")

    tasks = build_tasks(df)
    print(f"Built {len(tasks)} leave-one-out generation tasks")

    generated_df, error_df = await generate_leave_one_out(tasks)

    generated_df.to_csv(GENERATED_PATH, index=False)
    error_df.to_csv(ERRORS_PATH, index=False)

    print(f"Saved generated data to: {GENERATED_PATH}")
    print(f"Saved errors to: {ERRORS_PATH}")


if __name__ == "__main__":
    await main()
