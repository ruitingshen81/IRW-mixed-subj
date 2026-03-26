library(tidyverse)


d <- read_csv("data/raw/gilbert_meta_89_item_resp.csv")


person_item_long <- d |>
  filter(!is.na(resp)) |>
  select(id, item, item_text, resp) |>
  arrange(id, item) |> 
  distinct()

dup_pairs <- d |>
  count(id, item) |>
  filter(n > 1) |>
  select(id, item)
##some specific items have two responses for the same person, looked into those rows,
# apparently they have different "wave" values, i guess that may just be experiement design?
# but for our proj purpose i just removed all rows corresponding to those items to make it cleaner

person_item_long_clean <- person_item_long |>
  anti_join(dup_pairs, by = c("id", "item"))


write_csv(
  person_item_long,
  "data/processed/meta89/person-specific-itemresp-long.csv"
)


