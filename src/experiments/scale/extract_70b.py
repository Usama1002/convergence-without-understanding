# Scale validation: extract hidden states from 70B models
import os, json, time, torch, sys, numpy as np
os.environ['TORCHDYNAMO_DISABLE'] = '1'
sys.path.insert(0, '.')

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
if not hasattr(DynamicCache, 'get_max_length'):
    DynamicCache.get_max_length = lambda self: None

from src.data_loading import load_all_problems, format_prompt
from src.config import MAX_SEQ_LEN

MODELS_70B = [
    {"short_name": "LLaMA-3.1-70B", "hf_id": "meta-llama/Llama-3.1-70B-Instruct", "params_b": 70.0},
]

def extract_large_model(model_cfg, problems):
    hf_id = model_cfg["hf_id"]
    name = model_cfg["short_name"]
    print(f"\n{'='*60}\nExtracting: {name}\n{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Detect dimensions
    dummy = tokenizer("test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    n_layers_p1 = len(out.hidden_states)
    hidden_dim = out.hidden_states[0].shape[-1]
    del out
    print(f"Layers: {n_layers_p1-1}, Hidden dim: {hidden_dim}", flush=True)

    n_problems = len(problems)
    # Pre-decision only (post-decision too expensive for 70B)
    pre_decision = np.zeros((n_problems, n_layers_p1, hidden_dim), dtype=np.float32)

    for i, problem in enumerate(problems):
        user_prompt, sys_prompt = format_prompt(problem)
        messages = [{"role": "user", "content": (sys_prompt or "") + "\n" + user_prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = user_prompt
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for layer_idx, hs in enumerate(out.hidden_states):
            pre_decision[i, layer_idx] = hs[0, -1].float().cpu().numpy()
        del out
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  [{i+1}/{n_problems}]", flush=True)

    model_info = {
        "short_name": name, "hf_id": hf_id,
        "n_layers": n_layers_p1 - 1, "n_layers_plus_one": n_layers_p1,
        "hidden_dim": hidden_dim, "n_problems": n_problems,
    }

    del model
    torch.cuda.empty_cache()
    return pre_decision, model_info

if __name__ == "__main__":
    problems = load_all_problems()
    for model_cfg in MODELS_70B:
        name = model_cfg["short_name"]
        states, info = extract_large_model(model_cfg, problems)

        hs_dir = "data/hidden_states"
        os.makedirs(hs_dir, exist_ok=True)
        # No post-decision pass at 70B scale: NaN marks the states as
        # missing (exp10 skips them); a pre copy would compare pre vs pre.
        np.savez_compressed(
            f"{hs_dir}/{name}.npz",
            pre_decision=states,
            post_decision=np.full_like(states, np.nan),
        )
        with open(f"{hs_dir}/{name}_info.json", "w") as f:
            json.dump(info, f, indent=2)

        size_mb = os.path.getsize(f"{hs_dir}/{name}.npz") / (1024*1024)
        print(f"Saved {name}: {size_mb:.0f} MB", flush=True)
