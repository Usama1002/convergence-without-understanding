# Scale validation: evaluate 70B models
import os, json, time, torch, sys
os.environ['TORCHDYNAMO_DISABLE'] = '1'
sys.path.insert(0, '.')

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

from src.data_loading import load_all_problems, format_prompt
from src.evaluation import evaluate_response

MODELS_70B = [
    {"short_name": "LLaMA-3.1-70B", "hf_id": "meta-llama/Llama-3.1-70B-Instruct", "params_b": 70.0},
]

def evaluate_large_model(model_cfg, problems, max_new_tokens=256):
    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16,
        device_map="auto",  # Splits across both GPUs
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded across devices: {model.hf_device_map}", flush=True)

    results = []
    for i, problem in enumerate(problems):
        user_prompt, sys_prompt = format_prompt(problem)
        messages = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
        messages.append({"role": "user", "content": user_prompt})

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_time = time.time() - t0

        response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        eval_result = evaluate_response(response, problem["gold_answer"], problem["task_type"])

        results.append({
            "problem_id": problem["problem_id"],
            "domain": problem["domain"],
            "model": name,
            "correct": eval_result["correct"],
            "extracted_answer": eval_result["extracted_answer"],
            "gold_answer": problem["gold_answer"],
            "raw_response": response[:500],
            "task_type": problem["task_type"],
            "gen_time_s": gen_time,
        })

        if (i + 1) % 50 == 0:
            correct = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{len(problems)}] Accuracy so far: {correct}/{i+1} ({correct/(i+1)*100:.1f}%)", flush=True)

    del model
    torch.cuda.empty_cache()

    correct = sum(1 for r in results if r["correct"])
    print(f"\nFinal accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)", flush=True)
    return results

if __name__ == "__main__":
    problems = load_all_problems()
    print(f"Loaded {len(problems)} problems")

    for model_cfg in MODELS_70B:
        results = evaluate_large_model(model_cfg, problems)
        out_path = f"data/evaluations/{model_cfg['short_name']}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_path}")
