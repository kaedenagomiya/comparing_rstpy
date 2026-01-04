import time
import argparse
from llama_cpp import Llama

def format_qwen_prompt(user_message):
    """Qwen2.5の標準チャットフォーマット"""
    formated_prompt = f"<|im_start|>system\nYou are a helpful coding assistant.<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
    return formated_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../model/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    
    model_path = args.model_path
    prompt = format_qwen_prompt(args.prompt)

    # Load model
    load_start = time.perf_counter()
    llm = Llama(
            model_path=model_path,
            n_ctx=32768,
            #8192,16384  # コンテキスト長（必要に応じて調整）
            n_threads=8,  # CPUスレッド数（環境に応じて調整）
            n_gpu_layers=0,  # CPU推論の場合は0
            verbose=False
            )
    load_time = time.perf_counter() - load_start

    # Inference
    infer_start = time.perf_counter()
    result = llm(
                prompt,
                max_tokens=2048,
                #max_tokens=512,  # 生成する最大トークン数
                temperature=0.7,  # 温度パラメータ（0.0-1.0、低いほど決定的）
                top_p=0.95,  # Top-pサンプリング
                echo=False,  # プロンプトを出力に含めない
                stop=["<|im_end|>", "<|im_start|>"]  # Qwenの停止トークン
                #stop=["<|im_end|>", "###", "\n\n\n"]  # 停止条件
                )
    infer_time = time.perf_counter() - infer_start

    """
    llm = Llama(
        model_path=args.model_path,
        n_ctx=32768,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
        chat_format="chatml"  # Qwen用のフォーマットを指定
    )

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt}
        ],
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95
    )

    print(result["choices"][0]["message"]["content"])
    """

    """
    # Streaming inference
    print("\n--- Streaming Inference ---")
    stream = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        stream=True
    )

    for output in stream:
        text = output["choices"][0]["text"]
        print(text, end="", flush=True)

    print("\n")
    """

    # Evaluation
    tokens = result["usage"]["completion_tokens"]
    tokens_per_sec = tokens / infer_time
    # If finish_reason is "length", reach to max_tokens.
    # If finish_reason is "stop", match stop condition. 
    finish_reason = result["choices"][0]["finish_reason"]
    output_text = result["choices"][0]["text"].strip()
    print(
    f'''=== Python LLM Inference ===
    Model load time: {load_time:.2f} sec
    Inference time: {infer_time:.2f} sec
    Tokens generated: {tokens}
    Tokens/sec: {tokens_per_sec:.2f}
    Finish Reason: {finish_reason}
    Output:
    \t{output_text}''')



if __name__ == "__main__":
    main()
