use std::{collections::HashMap, path::PathBuf};

use anyhow::{Error, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use super::{constants, ModelConfig, NERModel, DTYPE};

/// Download model from HF
pub fn get_model_path(model: String) -> (PathBuf, PathBuf, PathBuf) {
    let api = Api::new().unwrap();
    let repo = api.model(model);

    (
        repo.get("model.safetensors").unwrap_or(PathBuf::default()),
        repo.get("tokenizer.json").unwrap_or(PathBuf::default()),
        repo.get("config.json").unwrap_or(PathBuf::default()),
    )
}

/// Get the model
pub fn get_ner_model() -> Result<(NERModel, Tokenizer)> {
    let (weights_path, tokenizer_path, config_path) =
        get_model_path("blaze999/Medical-NER".to_string());

    // Read and parse config file
    let config_str = std::fs::read_to_string(config_path).expect("Failed to read config file");
    let config: ModelConfig = serde_json::from_str(&config_str).expect("Failed to parse config");

    // Read tokenizer
    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("errors in tokenizer");
    // Load the model weights
    let device = candle_core::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &Device::Cpu)? };

    // Initialize label mappings from constants
    let mut id2label = HashMap::new();
    let mut label2id = HashMap::new();

    for (id, label) in constants::ID_TO_LABEL.entries() {
        id2label.insert(id.to_string(), label.to_string());
    }

    for (label, &id) in constants::LABEL_TO_ID.entries() {
        label2id.insert(label.to_string(), id);
    }

    // Update config with label mappings
    let mut config = config;
    config.id2label = id2label;
    config.label2id = label2id;
    //model
    let model = NERModel::load(vb, &config)?;
    // Return the model components
    Ok((model, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model_path() {
        let (model_path, tokenizer_path, config_path) =
            get_model_path("blaze999/Medical-NER".to_string());

        // Check that paths end with expected filenames
        assert!(model_path.ends_with("model.safetensors"));
        assert!(tokenizer_path.ends_with("tokenizer.json"));
        assert!(config_path.ends_with("config.json"));

        // Check that files exist
        assert!(model_path.exists());
        assert!(tokenizer_path.exists());
        assert!(config_path.exists());
    }

    #[test]
    fn test_get_model_path_invalid_model() {
        let (model_path, tokenizer_path, config_path) =
            get_model_path("non-existent-model".to_string());

        // Should return default paths for invalid model
        assert_eq!(model_path, PathBuf::default());
        assert_eq!(tokenizer_path, PathBuf::default());
        assert_eq!(config_path, PathBuf::default());
    }
}
