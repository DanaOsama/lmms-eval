import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

# Task link on HF: https://huggingface.co/datasets/howard-hou/OCR-VQA

def ocrvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


# From textvqa
def ocrvqa_process_results(doc, result):
    breakpoint()
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(results) == len(doc["questions"]), (
        f"Expected {len(doc['questions'])} results, but got {len(results)}."
    )
    # resAns = eval_ai_processor(result[0])
    # accuracy = 0

    processed_answers = [eval_ai_processor(ans) for ans in results]
    gt_answers = [eval_ai_processor(ans) for ans in doc["answers"]]

    accuracies = []
    for pred, gt in zip(processed_answers, gt_answers):
        matching_gt = [ans for ans in gt_answers if ans == pred]
        accuracy = min(1, len(matching_gt) / 3)
        accuracies.append(accuracy)
    
    # Compute a single accuracy metric for the entire image
    image_accuracy = statistics.mean(accuracies) if accuracies else 0


    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answers": processed_answers,
        },
    }


def ocrvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    ocr_ref = ""
    if lmms_eval_specific_kwargs:
        if "pre_prompt" in lmms_eval_specific_kwargs:
            pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        if "post_prompt" in lmms_eval_specific_kwargs:
            post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        if "ocr" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["ocr"]:
            ocr_ref = f"\nReference OCR token: {', '.join(doc['ocr_tokens'])}"
    # return f"{pre_prompt}{doc['questions'][0].capitalize()}{ocr_ref}{post_prompt}"

    # Since OCRVQA has multiple questions for one image, we join them.
    # Format all questions
    formatted_questions = "\n".join([f"Q{i+1}: {q.capitalize()}" for i, q in enumerate(doc["questions"])])

    return f"{pre_prompt}\n{formatted_questions}{ocr_ref}{post_prompt}"


def ocrvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"ocrvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    # print(f"Submission file saved to {path}")
    eval_logger.info(f"Submission file saved to {path}")
