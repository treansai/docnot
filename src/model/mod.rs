use std::{collections::HashMap, path::PathBuf};

use anyhow::{Ok, Result};
use candle_core::{DType, Result as CdlResult, Tensor, D};
use candle_nn::{embedding, linear, Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder};
use constants::{ID_TO_LABEL, LABEL_TO_ID};
use hf_hub::{api::sync::Api, Repo};
use serde::{Deserialize, Serialize};
use tokenizers::{tokenizer, PaddingParams, PaddingStrategy, Tokenizer};
pub mod build;
pub mod constants;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PosAttType {
    P2C,
    C2P,
}

#[derive(Deserialize)]
pub(crate) struct ModelConfig {
    hidden_dropout_prob: f32,
    hidden_size: usize,
    id2label: HashMap<String, String>,
    label2id: HashMap<String, usize>,
    layer_norm_eps: f64,
    relative_attention: bool,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    vocab_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_dropout_prob: 0.1,
            hidden_size: 768,
            id2label: HashMap::new(),
            label2id: HashMap::new(),
            layer_norm_eps: 1e-07,
            max_position_embeddings: 512,
            relative_attention: true,
            type_vocab_size: 0,
            vocab_size: 128100,
        }
    }
}

pub(crate) struct NERModel {
    word_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    classifier: Linear,
    span: tracing::Span,
}

impl NERModel {
    pub fn new(
        word_embeddings: Embedding,
        layer_norm: LayerNorm,
        dropout: Dropout,
        classifier: Linear,
    ) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "NERModel");
        Self {
            word_embeddings,
            layer_norm,
            dropout,
            classifier,
            span,
        }
    }

    pub fn load(vb: candle_nn::VarBuilder, config: &ModelConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("deberta.embeddings.word_embeddings"),
        )?;

        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("deberta.embeddings.LayerNorm"),
        )?;

        let dropout = Dropout::new(config.hidden_dropout_prob);

        let classifier = linear(
            config.hidden_size,
            config.label2id.len(),
            vb.pp("classifier"),
        )?;

        Ok(Self::new(word_embeddings, layer_norm, dropout, classifier))
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Get embeddings for input tokens
        let hidden_states = self.word_embeddings.forward(input_ids)?;

        // Apply layer norm and dropout
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, true)?;

        // Apply classification layer
        let logits = self.classifier.forward(&hidden_states)?;

        Ok(logits)
    }

    pub fn predict_sentence(&self, sentence: &str, tokenizer: &Tokenizer) -> Result<NERAnalysis> {
        let encoding = tokenizer.encode(sentence, true).unwrap();
        let tokens = encoding.get_tokens();
        let input_ids = encoding.get_ids();

        // Covert to tensor
        let input_tensor = Tensor::new(input_ids, &candle_core::Device::Cpu)?.unsqueeze(0)?;

        // Get model predictions
        let logits = self.forward(&input_tensor)?;

        // get proba
        let proba = candle_nn::ops::softmax(&logits, D::Minus1)?;

        // Get the highest prob for each token
        let pred = proba.argmax(D::Minus1)?;
        let pred_scores = proba.max(D::Minus1)?;

        // Convert predictions to labels
        let pred_array = pred.squeeze(0)?.to_vec1::<u32>()?;
        let scores = pred_scores.squeeze(0)?.to_vec1::<f32>()?;

        // result vec
        let mut res = Vec::new();

        for ((token, &pred_id), score) in tokens.iter().zip(pred_array.iter()).zip(scores.iter()) {
            let label = ID_TO_LABEL
                .get(&pred_id.to_string())
                .unwrap_or(&"0")
                .to_string();

            res.push(TokenPrediction {
                token: token.clone(),
                label,
                score: *score,
                start: None,
                end: None,
            });
        }
        Ok(NERAnalysis {
            text: sentence.to_string(),
            tokens: res,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenPrediction {
    token: String,
    label: String,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NERAnalysis {
    text: String,
    tokens: Vec<TokenPrediction>,
    // entities: Vec<Entity>,
}

// #[derive(Debug, Serialize, Deserialize)]
// pub struct Entity {
//     text: String,
//     label: String,
//     start: usize,
//     end: usize,
//     score: f32,
// }
//Result<Vec<TokenPrediction>
