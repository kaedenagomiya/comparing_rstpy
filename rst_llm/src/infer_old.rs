use std::path::{PathBuf};
use std::time::Instant;
use std::fs::File;
use std::io::BufReader;
use anyhow::{Result, anyhow};

use candle_core::{Device, Tensor};
use candle_transformers::models::llama::{LlamaConfig, LlamaEosToks};
//use candle_transformers::models::quantized_llama::model::Model;
use candle_transformers::models::quantized_llama::{ModelWeights, QuantizedLlama}; 
//use candle_transformers::models::quantized_llama::ModelWeights;
use candle_core::quantized::gguf_file::Content;
use tokenizers::Tokenizer;
use serde_json;

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
    // setting path
    let model_dir = PathBuf::from("../model");
    let model_path = model_dir.join("elyza-japanese-CodeLlama-7b-instruct-q4_k_m.gguf");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let config_path = model_dir.join("config.json");

    // device type
    let device = Device::Cpu;

    // load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Tokenizer load error: {}", e))?;

    // load model setting
    let config_data = std::fs::read(&config_path)?;
    let llama_config: LlamaConfig = serde_json::from_slice(&config_data)?;
    let config = llama_config.into_config(false);
    //let eos_token_id = config.eos_token_id.unwrap_or(2); // 安全にデフォルト値を設定
    let eos_token_id = match config.eos_token_id.clone().unwrap_or(LlamaEosToks::Single(2)) { 
        LlamaEosToks::Single(id) => id,
        LlamaEosToks::Multiple(ids) => ids[0],
    };


    // load model
    //let model = Model::load(&model_path, &config, &device)?;
    let file = File::open(&model_path)?;
    let mut reader = BufReader::new(file);
    let content = Content::read(&mut reader)?;
    let weights = ModelWeights::from_gguf(content, &mut reader, &device)?;
    //let logits = weights.forward(&input_tensor, past_kv_cache)?;
    let model = QuantizedLlama::load(&weights, &config)?;

    // format prompt
    let prompt = format_codellama_prompt("Rustはどんなプログラミング言語ですか？", None);

    // Tokenize
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow!("Tokenize error: {}", e))?;
    let mut input_ids = encoding.get_ids().to_vec();

    // inference loop（greedy decoding）
    let max_tokens = 100;
    let start = Instant::now();
    for _ in 0..max_tokens {
        let input = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?; // [1, seq_len]
        let logits = model.forward(&input, input_ids.len() - 1)?; // 最後の位置だけ
        let logits = logits.squeeze(0)?; // [vocab_size]
        let next_token = logits.argmax(candle_core::DType::U32)?;
        let token_id = next_token.to_scalar::<u32>()?;
        if token_id == eos_token_id {
            break;
        }
        input_ids.push(token_id);
    }
    let duration = start.elapsed();

    // decode and print
    let decoded = tokenizer.decode(&input_ids, true)
        .map_err(|e| anyhow!("Decode error: {}", e))?;
    let tokens_generated = input_ids.len();

    println!("\nOutput:\n{}", decoded);
    println!("\nInfer Time: {:?} sec", duration);
    println!("Mean token generated time: {:?} sec", duration / tokens_generated as u32);

    Ok(())
}

