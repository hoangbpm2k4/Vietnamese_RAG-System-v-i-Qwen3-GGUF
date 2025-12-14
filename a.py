from llama_cpp import Llama

# đổi sang q4_k_m.gguf nếu bạn đã quant
llm = Llama(
    model_path=r"qwen3_06b.gguf",
    n_ctx=1024,  # giảm nếu máy yếu
    verbose=True,
)

prompt = (
    "<|im_start|>system\n"
    "<|im_start|>user\n"
    "hôm nay tôi buồn quá\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

out = llm(prompt, max_tokens=32768, temperature=0.7, top_k=40, top_p=0.95, seed= 42)
print(out["choices"][0]["text"])
