//! Native powerset + ResNet34 stage loader. The stub below keeps the name
//! resolvable when the kernel features are off.

use super::{ConfigError, StagePair};
#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
use crate::embedder::Embedder;
use crate::models::ModelRegistry;
use crate::pipeline_v2::config::PipelineConfig;
#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
use crate::segmentation::Segmenter;

#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
pub(super) fn build_native_stages(
    registry: &ModelRegistry,
    config: &PipelineConfig,
) -> Result<StagePair, ConfigError> {
    tracing::info!("native kernels: powerset_int8 + resnet34_int8");
    let seg_path = registry
        .ensure("powerset_int8")
        .map_err(|e| ConfigError::Load {
            model_id: "powerset_int8",
            source: Box::new(e),
        })?;
    let segmenter: Box<dyn Segmenter> = Box::new(
        crate::segmentation::PowersetNative::from_onnx_path(&seg_path).map_err(|e| {
            ConfigError::Load {
                model_id: "powerset_int8",
                source: Box::new(e),
            }
        })?,
    );
    if let Some(id) = config.experimental.embedder_model.as_deref() {
        let embedder = load_native_embedder_override(registry, config, id)?;
        return Ok((segmenter, embedder));
    }
    let emb_path = registry
        .ensure("resnet34_int8")
        .map_err(|e| ConfigError::Load {
            model_id: "resnet34_int8",
            source: Box::new(e),
        })?;
    let embedder: Box<dyn Embedder> = Box::new(
        crate::embedder::ResNet34Native::from_onnx_path(&emb_path).map_err(|e| {
            ConfigError::Load {
                model_id: "resnet34_int8",
                source: Box::new(e),
            }
        })?,
    );
    Ok((segmenter, embedder))
}

#[cfg(all(feature = "segmenter-native", feature = "embedder-native"))]
fn load_native_embedder_override(
    #[cfg_attr(
        not(all(feature = "backend-tract", feature = "embedder")),
        allow(unused_variables)
    )]
    registry: &ModelRegistry,
    #[cfg_attr(
        not(all(feature = "backend-tract", feature = "embedder")),
        allow(unused_variables)
    )]
    config: &PipelineConfig,
    id: &str,
) -> Result<Box<dyn Embedder>, ConfigError> {
    let (model_id, dim): (&'static str, usize) = match id {
        "cam_pp_int8" => ("cam_pp_int8", 512),
        "cam_pp_fp32" => ("cam_pp_fp32", 512),
        other => {
            return Err(ConfigError::UnknownModel {
                model_id: other.to_string(),
            });
        }
    };
    #[cfg(all(feature = "backend-tract", feature = "embedder"))]
    {
        tracing::info!("native powerset + tract embedder override {model_id} ({dim}-d)");
        let path = registry.ensure(model_id).map_err(|e| ConfigError::Load {
            model_id,
            source: Box::new(e),
        })?;
        let embedder = crate::embedder::CamPlusPlusExtractor::new(
            &path,
            dim,
            config.embedder_pool_size.max(1),
            config.execution_provider,
        )
        .map_err(|e| ConfigError::Load {
            model_id,
            source: Box::new(e),
        })?;
        Ok(Box::new(embedder))
    }
    #[cfg(not(all(feature = "backend-tract", feature = "embedder")))]
    {
        let _ = dim;
        Err(ConfigError::Load {
            model_id,
            source: Box::new(std::io::Error::other(
                "CAM++ embedder override requires --features cli,backend-tract",
            )),
        })
    }
}

#[cfg(not(all(feature = "segmenter-native", feature = "embedder-native")))]
pub(super) fn build_native_stages(
    _registry: &ModelRegistry,
    _config: &PipelineConfig,
) -> Result<StagePair, ConfigError> {
    unreachable!("native stage loader compiled out")
}
