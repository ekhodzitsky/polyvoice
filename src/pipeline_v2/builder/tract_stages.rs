//! Tract stage loader.
//!
//! `infer` does not compile unless `backend-tract` is on, so this path always
//! loads the tract graphs. Shipping INT8 powerset and ResNet34 are not used.

use super::{ConfigError, StagePair};
use crate::embedder::Embedder;
use crate::models::ModelRegistry;
use crate::pipeline_v2::config::PipelineConfig;
use crate::segmentation::Segmenter;

pub(super) fn build_onnx_stages(
    registry: &ModelRegistry,
    config: &PipelineConfig,
) -> Result<StagePair, ConfigError> {
    let ep = config.execution_provider;
    let pool = config.embedder_pool_size;
    let mut seg_cfg = crate::segmentation::PowersetConfig::default();
    seg_cfg.aggregation.binarization = config.experimental.binarization;
    seg_cfg.pool_size = pool;
    tracing::info!("tract backend: loading powerset_fp32_tract (shipping powerset unsupported)");
    let segmenter_path = registry
        .ensure("powerset_fp32_tract")
        .map_err(|e| ConfigError::Load {
            model_id: "powerset_fp32_tract",
            source: Box::new(e),
        })?;
    let segmenter: Box<dyn Segmenter> = Box::new(
        crate::segmentation::PowersetSegmenter::with_config(&segmenter_path, seg_cfg, ep).map_err(
            |e| ConfigError::Load {
                model_id: "powerset_fp32_tract",
                source: Box::new(e),
            },
        )?,
    );
    tracing::info!("tract backend: loading FP32 wespeaker_resnet34 (INT8 unsafe under tract)");
    let embedder_path = registry
        .ensure("wespeaker_resnet34")
        .map_err(|e| ConfigError::Load {
            model_id: "wespeaker_resnet34",
            source: Box::new(e),
        })?;
    let embedder: Box<dyn Embedder> = Box::new(
        crate::embedder::ResNet34Adapter::new(&embedder_path, pool, ep).map_err(|e| {
            ConfigError::Load {
                model_id: "wespeaker_resnet34",
                source: Box::new(e),
            }
        })?,
    );
    Ok(StagePair {
        segmenter,
        embedder,
        segmenter_id: "powerset_fp32_tract",
        embedder_id: "wespeaker_resnet34",
    })
}
