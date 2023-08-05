"""
Extract logical CoT order from the input prompts, using the OpenAI API.
"""
import argparse
import datetime
import json
import os
import re
import sys

import openai
from csv import DictReader
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


regex = re.compile(r"\d+\.\s")


def wait_one_n_mins(n_mins=1):
    """
    Timeout for the OpenAI API.
    """
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=n_mins)
    while True:
        if datetime.datetime.now() >= endTime:
            break


def read_csv_in_dict(csv_path):
    """
    Read a csv file as a list of dictionaries.
    """
    with open(csv_path, "r") as f:
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)
    return list_of_dict


@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(10))
def run_chatgpt(model, temp, meta_prompt, max_tokens):
    """
    Run the ChatGPT model on the input prompt.
    """
    # Define the parameters for the text generation
    completions = openai.Completion.create(
        engine=model,
        prompt=meta_prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=temp,
    )
    gen_prompt = completions.choices[0].text.strip().lower()
    # Print the generated text
    print("The meta prompt is --> ", meta_prompt)
    print("LLM output is --> ", gen_prompt)
    return gen_prompt


def parse_llm_output(chatgpt_out, in_context):
    """
    Parse the LLM output.
    """
    if in_context:
        # Find the line that starts with r:
        chatgpt_out = chatgpt_out.split("\n")
        chatgpt_out = [x for x in chatgpt_out if x.startswith("r:")]
        if len(chatgpt_out) == 0:
            return [""]
        chatgpt_out = chatgpt_out[0]
        # Get inside the brackets
        in_str = chatgpt_out.split("[")[1].split("]")[0]
        in_str = in_str.split(",")
        # Map each string to its indexing number (e.g. "t":"stool")
        in_dict = {
            obj_str.split(":")[0].strip(): obj_str.split(":")[1].strip()
            for obj_str in in_str
        }
        ordered_list = []
        # Get the highest key number
        if len(in_dict.keys()) == 1:
            max_key = 0
        else:
            max_key = max([int(k) for k in in_dict.keys() if k.isdigit()])
            # Generate the ordered list
            ordered_list = [in_dict[str(i)] for i in range(1, max_key + 1)]
        # Add the target
        ordered_list.append(in_dict["t"])

    else:
        # Parse each object name into a list
        # Also remove starting numbers
        ordered_list = chatgpt_out.split("\n")
        ordered_list = [regex.sub("", x) for x in ordered_list]
        ordered_list = [x.strip() for x in ordered_list]

    return ordered_list


def parse_args(args):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the chatbot")
    parser.add_argument(
        "--api-key",
        type=str,
        default="TOKEN",
        help="OpenAI API key",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="CSV_FILE.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=70,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output_map.json",
        help="Output file name",
    )
    parser.add_argument(
        "--in-context-prompt",
        type=str,
        default="prompts/meta_prompt",
        help="Output file name",
    )
    parser.add_argument(
        "--in-context",
        action="store_true",
        help="Whether to use in-context learning",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for the data",
    )
    args = parser.parse_args(args)
    args.output_file = os.path.abspath(args.output_file)
    args.in_context_prompt = os.path.abspath(args.in_context_prompt)
    return args

def preProcess_anchors(anchor_list):
    
    middle_parsing_info = anchor_list.split('(')[1:]
    # print(middle_parsing_info)
    middle_parsing_info = [x.split(')')[0].split(',')[-1][2:-1] for x in middle_parsing_info]
    anchors_names = []
    for i in range(len(middle_parsing_info)):
        #removing ' from the beginning and end of the string
        info = middle_parsing_info[i]
        if(len(info) > 0 and info[0] == "'"):
            info = info[1:]
        if(len(info) > 0 and info[-1] == "'"):
            info = info[:-1]
        anchor_name = '_'.join(info.split('_')[1:])
        anchors_names.append(anchor_name)
    return anchors_names

def main():
    """
    Run main order extraction using the OpenAI API.
    """
    # Parsing input arguments
    args = parse_args(sys.argv[1:])
    print("Input arguments: ", args)
    openai.api_key = args.api_key

    # Read dataset
    nr3d_lst_dict = read_csv_in_dict(csv_path=args.csv_path)

    if args.in_context:
        meta_prompt = open(args.in_context_prompt, "r").read().strip()
    else:
        meta_prompt = "given the following sentence give me the logical order of objects to reach to the target object:"

    # Iterating over Nr3D
    output_map = []
    # Get the indices, take offset into account
    # if args.num_samples == -1:
    n_range = list(range(len(nr3d_lst_dict)))
    # else:
    #     n_range = list(
    #         range(args.offset, min(args.offset + args.num_samples, len(nr3d_lst_dict)))
    #     )
    for i in tqdm(n_range):
        # Extracting Nr3D fields
        nr3d_utterance = nr3d_lst_dict[i]["utterance"]
        nr3d_assigment = nr3d_lst_dict[i]["assignmentid"]
        nr3d_anchors   = preProcess_anchors(nr3d_lst_dict[i]["entities"])
        nr3d_target    = nr3d_lst_dict[i]["instance_type"]
        
        nr3d_row = i
        nr3d_scene = nr3d_lst_dict[i]["stimulus_id"].split("-")[0]
        gpt_in = meta_prompt + ' "' + nr3d_utterance + ' ' + str(nr3d_anchors) + '" '

        # Handle API timeout
        chatgpt_out = None
        while chatgpt_out is None:
            try:
                chatgpt_out = run_chatgpt(
                    model="text-davinci-003",
                    temp=0,
                    meta_prompt=gpt_in,
                    max_tokens=args.max_tokens,
                )
            except:
                print("OpenAI server out! Will try again Don't worry :D")
                pass

            # Save the generated output
            output_entry = {
                "assignmentid": nr3d_assigment,
                "row": nr3d_row,
                "scene": nr3d_scene,
                "utterance": nr3d_utterance,
                "output": chatgpt_out,  # parse_llm_output(chatgpt_out, args.in_context),
            }
            output_map += [output_entry]
    # Dump to JSON
    json.dump(output_map, open(args.output_file, "w"), indent=4)


if __name__ == "__main__":
    main()
