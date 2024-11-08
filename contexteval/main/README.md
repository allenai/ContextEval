## *Contextualized Evaluations*: Taking the Guesswork Out of Language Model Evaluations


## Data Overview

* The queries used for our main experiments are sampled from 5 datasets and can be found at `data/all_data_latest_filtered.jsonl`.

* The autorater judgements for all three model pairs are available at `data/autorater_judgements`.

* The human judgements for all three model pairs are available at `data/human_judgements`.


## Code Overview

#### Data Downloads
1. Download all the queries from all 5 datasets (`ChatBot Arena`, `MTBench`, `AlpacaEval`, `ExpertQA`, `KIWI`).
   * `python3 main/download_data.py`
2. Filter only well-formed queries and sample a fixed number of queries from each dataset.
   * `python3 main/filter_queries.py`


#### Context Generation

Example scripts for running context generation are at `bash_scripts/run_context_generation.sh`.

1. Generate contexts for all queries.
   * `python3 main/generate_contexts.py`

2. Generate validation labels for generated contexts.
   * `python3 main/generate_context_validation.py`

3. Generate a single instance of context for each query (follow-up question with single answer).
   * `python3 main/generate_single_context.py`


#### Response Generation

* Example scripts for running response generation are at `bash_scripts/run_response_generation.sh`.

* Generate model responses with and without context.
   * `python3 main/generate_responses.py`. Use `--w_context=False` for context-agnostic generation and `--w_context=True` for context-aware generation.

#### Autorater Evaluation Judgements

* Example scripts for generating pairwise evaluation judgements are at `bash_scripts/run_eval_generation.sh`. 
    * For the setting `CtxGen-CtxEval`, you would need to provide responses that are generated with context and for `NoCtxGen-NoCtxEval` and `NoCtxGen-CtxEval`, you would need to provide responses generated without context. For `CtxGen-CtxEval` and `NoCtxGen-CtxEval`, you would set `W_CONTEXT=True` while you would set `W_CONTEXT=False` for `NoCtxGen-NoCtxEval`.

* Generate pairwise evaluation judgements using the following script.
   * `python3 main/generate_pairwise_evals.py`

#### Win Rate and Agreement Calculation

* Computing win rates and agreement based on autorater judgments
   * `python3 main/compute_autorater_agreement.py`.

* Computing win rates and agreement based on human judgments
   * `python3 main/compute_human_agreement.py`.

#### What Contexts are "Default"?

* Example scripts for running this analysis are at `bash_scripts/default_response_analysis.sh`.

#### Which Contexts are Harder to Follow?

* Example scripts for running this analysis are at `bash_scripts/adapted_response_analysis.sh`.


#### Miscellaneous Scripts

* To generate types of each query based on the degree / type of underspecification, use the script `main/generate_query_types.py`.
* To compute the number of constraints (follow-up QAs) satisfied by each response, use the script `main/eval_num_constraints.py`.
* To codify autorater justifications, use the script `main/code_model_judgements.py` and to codify human justifications, use the script `main/code_human_judgements.py`.


### Citation

```
@inproceedings{malaviya2024contexteval,
   author = {Malaviya, Chaitanya and Chee Chang, Joseph and Roth, Dan and Iyyer, Mohit and Yatskar, Mark and Lo, Kyle},
   title = {Contextual Evaluations: Taking the Guesswork Out of Language Model Evaluations},
   journal = {arXiv},
   month = {November},
   year = {2024},
   url = "https://drive.google.com/file/d/1girNKIhwf63DO8KIPNs6v17hEcBaCOA4/view?usp=sharing"
}
```