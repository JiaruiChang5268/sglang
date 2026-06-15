import argparse
import json
import time

import answer_extraction
import eval_utils
import numpy as np
from datasets import load_dataset

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
from sglang.utils import dump_state_text


def build_prompt(question: str) -> str:
    return (
        question
        + "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


@sgl.function
def reasoning_gen(s, question: str):
    s += sgl.user(build_prompt(question))
    s += sgl.assistant(
        sgl.gen(
            "answer",
        )
    )


def convert_dataset(
    path: str,
    question_key: str,
    answer_key: str,
    num_tries: int,
    start_index: int,
    num_questions: int,
):
    import glob
    import os

    if os.path.exists(path):
        files = (
            sorted(glob.glob(os.path.join(path, "*.jsonl")))
            if os.path.isdir(path)
            else [path]
        )
        rows = load_dataset("json", data_files=files, split="train")
    else:
        rows = load_dataset(path)["train"]
    questions = []
    answers = []
    if start_index or num_questions is not None:
        end_index = len(rows) if num_questions is None else start_index + num_questions
        rows = rows.select(range(start_index, min(end_index, len(rows))))
    for data in rows:
        question = data[question_key]
        answer = data[answer_key]
        for _ in range(num_tries):
            questions.append({"question": question})
            answers.append({"answer": answer})
    return questions, answers


def main(args):
    # Select backend
    sgl.set_default_backend(select_sglang_backend(args))

    # Get dataset
    questions, answers = convert_dataset(
        args.data_path,
        args.question_key,
        args.answer_key,
        args.num_tries,
        args.start_index,
        args.num_questions,
    )

    # Run requests
    tic = time.perf_counter()
    states = reasoning_gen.run_batch(
        questions,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )
    latency = time.perf_counter() - tic

    # Extract results and record outcomes in a list.
    outcomes = []
    raw_results = []
    for i, state in enumerate(states):
        pred_answer = ""
        gt_answer = str(answers[i]["answer"])
        error = None
        output = state["answer"]
        meta_info = state.get_meta_info("answer") or {}
        try:
            extracted_answers = answer_extraction.extract_math_answer(
                questions[i]["question"], output, "limo"
            )
            pred_answer = (
                extracted_answers[-1]
                if isinstance(extracted_answers, list) and extracted_answers
                else extracted_answers
            )
            if pred_answer is None:
                pred_answer = ""
            is_correct = 1 if eval_utils.math_equal(pred_answer, gt_answer) else 0
        except Exception as e:
            print(f"Error extracting answer: {e}")
            error = str(e)
            is_correct = 0

        outcomes.append(is_correct)
        raw_results.append(
            {
                "prompt_id": i,
                "question": questions[i]["question"],
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "correct": bool(is_correct),
                "completion_tokens": meta_info.get("completion_tokens"),
                "meta_info": meta_info,
                "error": error,
                "output": output,
            }
        )
        if not is_correct:
            preview = repr(output[:200])
            print(
                f"[wrong] idx={i} pred={pred_answer!r} gt={gt_answer!r} "
                f"prompt_tokens={meta_info.get('prompt_tokens')} "
                f"completion_tokens={meta_info.get('completion_tokens')} "
                f"finish_reason={meta_info.get('finish_reason')} "
                f"output_preview={preview}"
            )

    # Calculate overall accuracy using numpy
    overall_accuracy = np.mean(outcomes)
    print(f"Overall Accuracy: {overall_accuracy}")

    # Calculate mean standard error over questions if num_tries >= 2
    if args.num_tries > 1:
        outcomes_np = np.array(outcomes).reshape(-1, args.num_tries)
        # Using sample standard deviation with ddof=1
        std_per_question = np.std(outcomes_np, axis=1, ddof=1)
        # Compute the standard error for each question: std / sqrt(num_tries)
        se_per_question = std_per_question / np.sqrt(args.num_tries)
        mean_se = se_per_question.mean()
        print(f"Mean Standard Error of Accuracy across questions: {mean_se}")
    else:
        mean_se = None
        print("Not enough samples per question to compute standard error.")

    # Calculate output throughput
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency
    print(f"Output throughput: {output_throughput} token/s")

    # Dump results
    dump_state_text(f"tmp_output_{args.backend}.txt", states)
    if args.raw_result_file:
        with open(args.raw_result_file, "w", encoding="utf-8") as fout:
            for row in raw_results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Raw per-sample results: {args.raw_result_file}")

    # Write results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "limo",
            "backend": args.backend,
            "latency": round(latency, 3),
            "overall_accuracy": round(overall_accuracy, 3),
            "mean_se_accuracy": round(mean_se, 3) if mean_se is not None else None,
            "num_requests": len(questions),
            "other": {
                "num_questions": len(questions),
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="GAIR/LIMO")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--num-tries", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    main(args)
