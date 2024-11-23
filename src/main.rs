use std::sync::Arc;

use api::{init_router, Context, Pipeline};
use candle_core::Device;
use model::build::get_ner_model;

mod api;
mod model;

#[tokio::main]
async fn main(){

    let (model, tokenizer) = get_ner_model().unwrap();
    let pipeline = Pipeline {
        model,
        tokenizer,
        device: Device::Cpu
    };
    
    let context = Context { pipeline};

    // cors
    let app = init_router(Arc::new(context));
    
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", "0.0.0.0", "9494"))
        .await
        .unwrap();

    axum::serve(listener, app.into_make_service())
    .await
    .unwrap();

}
