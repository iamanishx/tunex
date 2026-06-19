import argparse
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_arguments():
    p = argparse.ArgumentParser(description="Compare two models side-by-side.")
    p.add_argument("--model_a", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Path or HF ID of Model A (Instruct)")
    p.add_argument("--model_b", type=str, default="/data/axeai_m_0.1", help="Path or HF ID of Model B (AxeAI Merged)")
    p.add_argument("--greedy", action="store_true", default=True, help="Use greedy decoding for deterministic comparison")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (if not greedy)")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    return p.parse_args()

def generate_response(model, tokenizer, prompt, system_prompt, gen_kwargs):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        eos_ids = [151643, 151645]
        if tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)
            
        outputs = model.generate(
            **inputs,
            eos_token_id=eos_ids,
            **gen_kwargs
        )
        
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def main():
    args = parse_arguments()
    
    # Resolve Model B path
    model_b_path = args.model_b
    if not os.path.exists(model_b_path):
        model_b_path = "./axeai_m_0.1" if not os.path.exists("/data") else args.model_b
        
    print(f"Loading Model A: {args.model_a}")
    tokenizer_a = AutoTokenizer.from_pretrained(args.model_a, trust_remote_code=True)
    if tokenizer_a.pad_token is None:
        tokenizer_a.pad_token = "<|endoftext|>"
        
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading Model B: {model_b_path}")
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_path, trust_remote_code=True)
    if tokenizer_b.pad_token is None:
        tokenizer_b.pad_token = "<|endoftext|>"
        
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_path,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup generation configuration
    gen_kwargs = {
        "max_new_tokens": 512,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.85
        gen_kwargs["top_k"] = 50

    SYSTEM_PROMPT = "Tum ek helpful AI assistant ho. Tum Hinglish mein jawab dete ho, yaani Hindi ko English letters mein likhte ho. Agar user English mein puchhe toh bhi Hinglish mein jawab do."
    
    print("\n" + "="*70)
    print("Model Comparison Tool is ready!")
    print(f"Mode: {'Greedy (Deterministic)' if args.greedy else f'Sampling (Temp={args.temperature})'}")
    print("Type your prompt and press Enter. Type 'exit' to quit.")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not prompt:
            continue
        if prompt.lower() in ["exit", "quit"]:
            break
            
        print("\n" + "-"*35 + " GENERATING RESPONSES " + "-"*35)
        
        # Model A Response
        try:
            resp_a = generate_response(model_a, tokenizer_a, prompt, SYSTEM_PROMPT, gen_kwargs)
        except Exception as e:
            resp_a = f"Error generating from Model A: {e}"
            
        # Model B Response
        try:
            resp_b = generate_response(model_b, tokenizer_b, prompt, SYSTEM_PROMPT, gen_kwargs)
        except Exception as e:
            resp_b = f"Error generating from Model B: {e}"
            
        print(f"\n[MODEL A] (Base: {args.model_a}):")
        print("*" * 40)
        print(resp_a)
        print("*" * 40)
        
        print(f"\n[MODEL B] (Fine-tuned: {args.model_b}):")
        print("*" * 40)
        print(resp_b)
        print("*" * 40)
        print("\n" + "="*70)

if __name__ == "__main__":
    main()
