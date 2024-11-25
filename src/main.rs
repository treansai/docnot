use std::sync::Arc;

use anyhow::Result;
use api::{init_router, Context, Pipeline};
use candle_core::Device;
use model::build::get_ner_model;

mod api;
mod model;

#[tokio::main]
async fn main() -> Result<()> {
    let (model, tokenizer) = get_ner_model().unwrap();
    let pipeline = Pipeline::new(model, Device::Cpu, tokenizer);

    let context = Context { pipeline };

    let app = init_router(Arc::new(context));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 9494));
    println!("Server starting on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind to address: {}", e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}
