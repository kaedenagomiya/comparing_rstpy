use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor, DType};
use candle_core::quantized::gguf_file::Content;
use candle_transformers::models::llama::{LlamaConfig, LlamaEosToks};
use candle_transformers::models::quantized_llama::ModelWeights;
use tokenizers::Tokenizer;

fn format_codellama_prompt(user_prompt: &str, system_prompt: Option<&str>) -> String {
    let bos_token = "<s>";
    let b_inst = "[INST]";
    let e_inst = "[/INST]";
    let b_sys = "<<SYS>>\n";
    let e_sys = "\n<</SYS>>\n\n";

    let system = system_prompt.unwrap_or("あなたは誠実で優秀な日本人のアシスタントです。");
    format!(
        "{bos} {b_inst} {b_sys}{sys}{e_sys}{user} {e_inst}",
        bos = bos_token,
        b_inst = b_inst,
        b_sys = b_sys,
        sys = system,
        e_sys = e_sys,
        user = user_prompt,
        e_inst = e_inst
    )
}

fn main() -> Result<()> {
    // モデルとトークナイザのパス
    let model_dir = PathBuf::from("../model/elyza-japanese-CodeLlama-7b-instruct-q4_K_M");
    let model_path = model_dir.join("elyza-japanese-CodeLlama-7b-instruct-q4_k_m.gguf");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");

    // デバイス設定
    let device = Device::Cpu;
    let _dtype = DType::BF16;

    // トークナイザ読み込み
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Tokenizer load error: {}", e))?;

    // モデル設定読み込み
    let config_data = std::fs::read(&config_path)?;
    let llama_config: LlamaConfig = serde_json::from_slice(&config_data)?;
    let config = llama_config.into_config(false);
    let eos_token_id = match config.eos_token_id.clone().unwrap_or(LlamaEosToks::Single(2)) {
        LlamaEosToks::Single(id) => id,
        LlamaEosToks::Multiple(ids) => ids[0],
    };

    // GGUFファイル読み込み
    let file = File::open(&model_path)?;
    let mut reader = BufReader::new(file);
    let content = Content::read(&mut reader)?;
    let mut weights = ModelWeights::from_gguf(content, &mut reader, &device)?;

    // プロンプト整形
    let prompt = format_codellama_prompt("Rustはどんなプログラミング言語ですか？", None);

    // トークナイズ
    let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow!("Tokenize error: {}", e))?;
    let mut input_ids = encoding.get_ids().to_vec();

    // 推論ループ（greedy decoding）
    let max_tokens = 100;
    let start = Instant::now();
    for _ in 0..max_tokens {
        let input = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?; // [1, seq_len]
        let logits = weights.forward(&input, input_ids.len() - 1)?; // 最後の位置だけ
        let logits = logits.squeeze(0)?; // [vocab_size]
        let next_token = logits.argmax(0)?;
        let token_id = next_token.to_scalar::<u32>()?;
        if token_id == eos_token_id {
            break;
        }
        input_ids.push(token_id);
    }
    let duration = start.elapsed();

    // デコードして表示
    let decoded = tokenizer.decode(&input_ids, true).map_err(|e| anyhow!("Decode error: {}", e))?;
    let tokens_generated = input_ids.len();

    println!("\n出力:\n{}", decoded);
    println!("\n推論時間: {:?} 秒", duration);
    println!("平均トークン生成時間: {:?} 秒", duration / tokens_generated as u32);

    Ok(())
}

