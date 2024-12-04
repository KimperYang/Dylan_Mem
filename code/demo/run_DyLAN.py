import ast
import json
import os
import openai
import random
import sys
from prettytable import PrettyTable
from LLMLP import LLMLP
from utils import *

# openai.api_key =
# openai.api_base =
# openai.api_type =
# openai.api_version =

# Put your query here
QUERY = r"""What is the best soccor player"""

EXP_NAME = "trial_1"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

ACTIVATION = "listwise"
TYPE = "open-ended"
DIR_NAME = "trial"
GENERATION_MODE = "bias"
# Here are the roles of the participants in the LLM-agent collaboration
# See prompt_lib.ROLE_MAP for the full list of roles
ROLES = ["Assistant", "Assistant"]

def set_rd_seed(seed):
    random.seed(seed)

def main():
    set_rd_seed(0)
    assert len(ROLES) > 0

    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL, GENERATION_MODE)

    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(QUERY)
    imp_score = llmlp.backward(res)
    imp_score = [[imp_score[idx] for idx in range(len(ROLES)*rid, len(ROLES)*(rid+1))] for rid in range(3)]

    pt = PrettyTable()
    pt.add_column("Round", ROLES)
    for rid in range(3):
        responses = [(completions[idx][rid] if completions[idx][rid] is not None else "No response.") for idx in range(len(ROLES))]
        pt.add_column(str(rid+1), responses, "l")

    print(r"Query: {}".format(QUERY))
    print(r"#API calls: {}".format(resp_cnt))
    print(r"Prompt Tokens: {}".format(prompt_tokens))
    print(r"Completion Tokens: {}".format(completion_tokens))
    print(pt)
    print(r"Final Answer: {}".format(res))
    print()
    print(r"Agent Importance Scores: {}".format([sum(imp_score[rid][idx] for rid in range(3)) for idx in range(len(ROLES))]))


if __name__ == "__main__":
    main()
