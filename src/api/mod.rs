use std::sync::Arc;

use anyhow::Ok;
use anyhow::Result;
use axum::http::Method;
use axum::{
    extract::{Path, State},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use candle_core::Device;
use serde_json::json;
use tokenizers::Tokenizer;
use tower_http::cors::{CorsLayer, Any};

use crate::model::NERModel;

pub struct Pipeline {
    pub model: NERModel,
    pub device: Device,
    pub tokenizer: Tokenizer,
}

pub struct Context {
    pub pipeline: Pipeline,
}

impl Pipeline {
    pub fn new(model: NERModel, device: Device, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            device,
            tokenizer,
        }
    }

    pub fn run(&self, document: String) -> Result<serde_json::Value> {
        let res = self
            .model
            .predict_sentence(&document, &self.tokenizer)
            .unwrap();
        Ok(json!(res))
    }
}

pub fn init_router(state: Arc<Context>) -> Router {

    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_origin(Any);

    Router::new()
        .route("/", get(base_handler))
        .route("/predict:text", get(get_analyse))
        .layer(cors)
        .with_state(state)
}

pub async fn base_handler() -> impl IntoResponse {
    Json("Hello word").into_response()
}

pub async fn get_analyse(
    State(context): State<Arc<Context>>,
    Path(text): Path<String>,
) -> impl IntoResponse {
    let res = context.pipeline.run(text).unwrap();
    Json(res)
}
