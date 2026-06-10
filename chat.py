import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter", type=str, default="/data/qwen_hinglish_lora")
    p.add_argument("--no_history", action="store_true", help="Disable conversation history (single-turn mode)")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding (ignores temperature)")
    return p.parse_args()

def main():
    args = parse_arguments()
    
    # Check if adapter path exists
    adapter_path = args.adapter
    if not os.path.exists(adapter_path):
        adapter_path = "./qwen_hindi_lora_final" if not os.path.exists("/data") else args.adapter
        
    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\n" + "="*50)
    print("Chat session started! Type your message and press Enter.")
    print("Commands:")
    print("  'exit' or 'quit' to end the session.")
    print("  'clear' to clear conversation history.")
    print("="*50 + "\n")
    SYSTEM_PROMPT = "Tum ek helpful AI assistant ho. Tum Hinglish mein jawab dete ho, yaani Hindi ko English letters mein likhte ho. Agar user English mein puchhe toh bhi Hinglish mein jawab do."
    
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat session.")
            break
            
        if user_input.lower() == "clear":
            history = []
            print("Conversation history cleared.")
            continue
            
        # Append user message to history
        if args.no_history:
            history = [{"role": "user", "content": user_input}]
        else:
            history.append({"role": "user", "content": user_input})
        
        # Prepend system prompt to messages sent to the model
        messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        
        # Format the input using the model's chat template
        inputs = tokenizer.apply_chat_template(
            messages_for_model,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate output
        with torch.no_grad():
            # Standard Qwen EOS token IDs: 151643 (<|endoftext|>) and 151645 (<|im_end|>)
            eos_ids = [151643, 151645]
            if tokenizer.eos_token_id not in eos_ids:
                eos_ids.append(tokenizer.eos_token_id)
                
            gen_kwargs = {
                "max_new_tokens": 512,
                "repetition_penalty": args.repetition_penalty,
                "eos_token_id": eos_ids,
            }
            if args.greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = 0.85
                gen_kwargs["top_k"] = 50

            outputs = model.generate(
                **inputs,
                **gen_kwargs
            )
            
        # Decode only the newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        print(f"\nModel: {response}")
        
        # Append assistant response to history
        if not args.no_history:
            history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
