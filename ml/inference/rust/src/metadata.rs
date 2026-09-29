//! Exact, graph-backed host metadata exceptions; never an operator-wide CPU fallback.
use crate::{Config, Error, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    ffi::CStr,
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Deserialize)]
pub(crate) struct Policy {
    schema_version: u32,
    kind: String,
    model_sha256: String,
    native_runtime_sha256: String,
    ort_version: String,
    optimization_level: String,
    input_shape: [u32; 3],
    pub(crate) embedding_dimension: usize,
    evidence: Evidence,
    cpu_nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
struct Evidence {
    optimized_graph_sha256: String,
    profile_sha256: String,
    profile_batches: Vec<usize>,
    audited_cpu_node_count: usize,
}

#[derive(Debug, Deserialize)]
struct Node {
    name: String,
    op_type: String,
    outputs: Vec<Output>,
    role: String,
    proof: Value,
}

#[derive(Debug, Deserialize)]
struct Output {
    dtype: String,
    max_elements: u64,
    max_bytes: u64,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub(crate) fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn runtime_version(path: &Path) -> Result<String> {
    // SAFETY: the caller explicitly supplies an ORT native library. The API-base
    // ABI is the same stable entry point used by ort's own dynamic loader. Keep
    // the library alive until its static C string has been copied into Rust.
    unsafe {
        let library = libloading::Library::new(path).map_err(|e| invalid(e.to_string()))?;
        let getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
            library
                .get(b"OrtGetApiBase")
                .map_err(|e| invalid(e.to_string()))?;
        let base = getter()
            .as_ref()
            .ok_or_else(|| invalid("null OrtApiBase"))?;
        if (base.GetApi)(24) != std::ptr::from_ref(ort::api()) {
            return Err(invalid(
                "requested runtime differs from process-global ORT runtime",
            ));
        }
        let version = (base.GetVersionString)();
        if version.is_null() {
            return Err(invalid("null ORT version"));
        }
        Ok(CStr::from_ptr(version)
            .to_str()
            .map_err(|e| invalid(e.to_string()))?
            .to_owned())
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn bytes_per_element(dtype: &str) -> Option<u64> {
    match dtype {
        "int64" => Some(8),
        "int32" | "float" => Some(4),
        "float16" => Some(2),
        "bool" => Some(1),
        _ => None,
    }
}

impl Policy {
    pub(crate) fn load(path: &Path, model: &Path, runtime: &Path, config: &Config) -> Result<Self> {
        let policy: Self = serde_json::from_reader(BufReader::new(File::open(path)?))
            .map_err(|e| invalid(format!("metadata policy JSON: {e}")))?;
        policy.validate()?;
        if policy.model_sha256 != sha256(model)? {
            return Err(invalid("metadata policy ONNX SHA256 mismatch"));
        }
        if policy.native_runtime_sha256 != sha256(runtime)? {
            return Err(invalid("metadata policy native runtime SHA256 mismatch"));
        }
        if policy.ort_version != runtime_version(runtime)? {
            return Err(invalid("metadata policy ORT version mismatch"));
        }
        if policy.input_shape != [3, config.size, config.size] {
            return Err(invalid("metadata policy input shape/config mismatch"));
        }
        Ok(policy)
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.kind != "audited_host_metadata"
            || self.ort_version != "1.24.4"
            || self.optimization_level != "all"
            || self.embedding_dimension == 0
            || self.input_shape[0] != 3
            || self.input_shape[1] == 0
            || self.input_shape[2] == 0
        {
            return Err(invalid("unsupported metadata policy contract"));
        }
        for hash in [
            &self.model_sha256,
            &self.native_runtime_sha256,
            &self.evidence.optimized_graph_sha256,
            &self.evidence.profile_sha256,
        ] {
            if !valid_hash(hash) {
                return Err(invalid("invalid policy SHA256"));
            }
        }
        if self.cpu_nodes.is_empty()
            || self.evidence.audited_cpu_node_count != self.cpu_nodes.len()
            || !self.evidence.profile_batches.contains(&1)
            || self.evidence.profile_batches.contains(&0)
        {
            return Err(invalid(
                "missing or inconsistent offline placement evidence",
            ));
        }
        let mut names = HashSet::new();
        for node in &self.cpu_nodes {
            if node.name.is_empty()
                || node.op_type.is_empty()
                || !names.insert(&node.name)
                || node.outputs.is_empty()
                || node.proof["dataflow_audited"] != true
            {
                return Err(invalid("invalid, duplicate or unaudited CPU node"));
            }
            for output in &node.outputs {
                let bytes = bytes_per_element(&output.dtype)
                    .ok_or_else(|| invalid("unsupported CPU output dtype"))?;
                if output.max_elements == 0
                    || output.max_elements.checked_mul(bytes) != Some(output.max_bytes)
                {
                    return Err(invalid("CPU output bound must agree in elements and bytes"));
                }
                match node.role.as_str() {
                    "integer_metadata"
                        if matches!(output.dtype.as_str(), "int64" | "int32" | "bool") => {}
                    "dimension_scalar"
                        if matches!(output.dtype.as_str(), "float" | "float16")
                            && output.max_elements == 1
                            && node.outputs.len() == 1
                            && node.proof["head_dimension"].as_u64() == Some(64)
                            && node.proof["derived_value"]
                                .as_f64()
                                .is_some_and(f64::is_finite) => {}
                    _ => {
                        return Err(invalid(
                            "CPU output does not satisfy declared metadata role",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn verify_profile(&self, profile: &Path) -> Result<Value> {
        let events: Vec<Value> =
            serde_json::from_reader(BufReader::new(File::open(profile)?)).map_err(|e| invalid(e.to_string()))?;
        let counts = self.verify_events(&events)?;
        Ok(
            serde_json::json!({"status":"accepted_audited_host_metadata", "all_cuda":false,
            "model_sha256":self.model_sha256,"native_runtime_sha256":self.native_runtime_sha256,
            "ort_version":self.ort_version,"optimization_level":self.optimization_level,
            "startup_batches":[1],"profile":profile,"profile_sha256":sha256(profile)?,
            "cpu_events":counts.0,"cuda_events":counts.1,"cpu_nodes_observed":counts.2,
            "offline_optimized_graph_sha256":self.evidence.optimized_graph_sha256,
            "offline_profile_sha256":self.evidence.profile_sha256,
            "offline_profile_batches":self.evidence.profile_batches,
            "graph_ancestry":"offline reviewed proof bound to model bytes; not re-derived by Rust",
            "quality_validation":"separate required gate; startup checks only shape/finiteness/nonzero norm"}),
        )
    }

    fn verify_events(&self, events: &[Value]) -> Result<(usize, usize, usize)> {
        self.validate()?;
        let allowed: HashMap<_, _> = self
            .cpu_nodes
            .iter()
            .map(|node| (node.name.as_str(), node))
            .collect();
        let mut seen = HashSet::new();
        let (mut cpu, mut cuda) = (0, 0);
        for event in events {
            let Some(name) = event["name"]
                .as_str()
                .and_then(|n| n.strip_suffix("_kernel_time"))
            else {
                if event["args"]["provider"].is_string() {
                    return Err(invalid(
                        "unrecognized provider event outside kernel profile format",
                    ));
                }
                continue;
            };
            let args = &event["args"];
            match args["provider"].as_str() {
                Some("CUDAExecutionProvider") => {
                    cuda += 1;
                    continue;
                }
                Some("CPUExecutionProvider") => cpu += 1,
                _ => return Err(invalid(format!("unknown/missing provider for {name}"))),
            }
            let node = allowed
                .get(name)
                .ok_or_else(|| invalid(format!("unreviewed CPU node {name}")))?;
            if args["op_name"] != node.op_type {
                return Err(invalid(format!("CPU operator mismatch: {name}")));
            }
            let outputs = args["output_type_shape"]
                .as_array()
                .ok_or_else(|| invalid("missing CPU tensor shapes"))?;
            if outputs.len() != node.outputs.len() {
                return Err(invalid(format!("CPU output count mismatch: {name}")));
            }
            let mut total_bytes = 0_u64;
            for (actual, rule) in outputs.iter().zip(&node.outputs) {
                let object = actual
                    .as_object()
                    .ok_or_else(|| invalid("invalid CPU output descriptor"))?;
                let shape = object
                    .get(&rule.dtype)
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid(format!("CPU dtype mismatch: {name}")))?;
                if object.len() != 1 {
                    return Err(invalid("ambiguous CPU output dtype"));
                }
                let elements = shape
                    .iter()
                    .try_fold(1_u64, |n, d| n.checked_mul(d.as_u64()?))
                    .ok_or_else(|| invalid("invalid/overflowed CPU output shape"))?;
                let bytes = elements
                    .checked_mul(
                        bytes_per_element(&rule.dtype).ok_or_else(|| invalid("unknown dtype"))?,
                    )
                    .ok_or_else(|| invalid("CPU tensor byte overflow"))?;
                if elements == 0 || elements > rule.max_elements || bytes > rule.max_bytes {
                    return Err(invalid(format!(
                        "CPU tensor exceeds audited bounds: {name}"
                    )));
                }
                total_bytes = total_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("CPU output byte overflow"))?;
            }
            let reported = args["output_size"]
                .as_u64()
                .or_else(|| args["output_size"].as_str().and_then(|s| s.parse().ok()));
            if reported != Some(total_bytes) {
                return Err(invalid(format!("CPU profile output byte mismatch: {name}")));
            }
            seen.insert(name);
        }
        if cuda == 0 {
            return Err(invalid("startup must execute CUDA"));
        }
        Ok((cpu, cuda, seen.len()))
    }
}

pub(crate) fn audit_directory() -> Result<PathBuf> {
    let base = std::env::var_os("ORT_POLICY_PROFILE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    fs::create_dir_all(&base)?;
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| invalid(e.to_string()))?
        .as_nanos();
    let directory = base.join(format!("lct-metadata-{}-{time}", std::process::id()));
    fs::create_dir(&directory)?;
    Ok(directory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_json() -> Value {
        json!({
        "schema_version":1,"kind":"audited_host_metadata","model_sha256":"0".repeat(64),
        "native_runtime_sha256":"1".repeat(64),"ort_version":"1.24.4","optimization_level":"all",
        "input_shape":[3,256,256],"embedding_dimension":384,
        "evidence":{"optimized_graph_sha256":"2".repeat(64),"profile_sha256":"3".repeat(64),
            "profile_batches":[1,17],"audited_cpu_node_count":2},
        "cpu_nodes":[
            {"name":"shape","op_type":"Gather","outputs":[{"dtype":"int64","max_elements":4,"max_bytes":32}],
             "role":"integer_metadata","proof":{"dataflow_audited":true}},
            {"name":"head","op_type":"Sqrt","outputs":[{"dtype":"float","max_elements":1,"max_bytes":4}],
             "role":"dimension_scalar","proof":{"dataflow_audited":true,"head_dimension":64,"derived_value":8.0}}
        ]})
    }

    fn fixture() -> Policy {
        serde_json::from_value(fixture_json()).unwrap()
    }

    #[test]
    fn policy_rejects_changed_model_and_runtime_before_loading_native_code() {
        let directory = audit_directory().unwrap();
        let model = directory.join("model.onnx");
        let runtime = directory.join("runtime.dll");
        let path = directory.join("policy.json");
        fs::write(&model, b"model fixture").unwrap();
        fs::write(&runtime, b"not executable").unwrap();
        let config: Config = serde_json::from_value(json!({
            "size":256,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5],"mode":"stretch"
        }))
        .unwrap();
        let mut value = fixture_json();
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            Policy::load(&path, &model, &runtime, &config)
                .unwrap_err()
                .to_string()
                .contains("ONNX SHA256 mismatch")
        );
        value["model_sha256"] = json!(sha256(&model).unwrap());
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            Policy::load(&path, &model, &runtime, &config)
                .unwrap_err()
                .to_string()
                .contains("native runtime SHA256 mismatch")
        );
        for file in [path, model, runtime] {
            fs::remove_file(file).unwrap();
        }
        fs::remove_dir(directory).unwrap();
    }

    fn events() -> Vec<Value> {
        vec![
            json!({"name":"shape_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"Gather","output_type_shape":[{"int64":[4]}],"output_size":"32"}}),
            json!({"name":"head_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"Sqrt","output_type_shape":[{"float":[]}],"output_size":"4"}}),
            json!({"name":"learned_matmul_kernel_time","args":{"provider":"CUDAExecutionProvider","op_name":"MatMul"}}),
        ]
    }

    #[test]
    fn exact_profile_accepts_metadata_scalars_and_folded_node_subset() {
        assert_eq!(fixture().verify_events(&events()).unwrap(), (2, 1, 2));
        assert_eq!(fixture().verify_events(&events()[1..]).unwrap(), (1, 1, 1));
    }

    #[test]
    fn profile_rejects_unknown_cpu_operator_dtype_size_and_provider() {
        for (key, value) in [
            ("name", json!("unreviewed_kernel_time")),
            ("op_name", json!("MatMul")),
            ("provider", json!("UnreviewedExecutionProvider")),
            ("output_type_shape", json!([{"float":[4]}])),
            ("output_type_shape", json!([{"int64":[5]}])),
            ("output_type_shape", json!([{"int64":[-1]}])),
            ("output_size", json!("31")),
        ] {
            let mut changed = events();
            if key == "name" {
                changed[0][key] = value;
            } else {
                changed[0]["args"][key] = value;
            }
            assert!(fixture().verify_events(&changed).is_err(), "accepted {key}");
        }
        assert!(fixture().verify_events(&events()[..2]).is_err());
        let mut learned_cpu = events();
        learned_cpu.push(json!({"name":"learned_matmul_kernel_time","args":{"provider":"CPUExecutionProvider","op_name":"MatMul","output_type_shape":[{"float":[1]}],"output_size":"4"}}));
        assert!(fixture().verify_events(&learned_cpu).is_err());
        let mut hidden_provider = events();
        hidden_provider[0]["name"] = json!("unrecognized_event_format");
        assert!(fixture().verify_events(&hidden_provider).is_err());
    }

    #[test]
    fn policy_rejects_unreviewed_roles_bounds_and_runtime_contracts() {
        let mut policy = fixture();
        policy.cpu_nodes[0].proof["dataflow_audited"] = json!(false);
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.cpu_nodes[1].proof["head_dimension"] = json!(128);
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.cpu_nodes[1].outputs[0].max_elements = 2;
        policy.cpu_nodes[1].outputs[0].max_bytes = 8;
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.cpu_nodes[0].role = "image_values".into();
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.ort_version = "1.24.5".into();
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.optimization_level = "basic".into();
        assert!(policy.validate().is_err());
        let mut policy = fixture();
        policy.cpu_nodes[1].name = "shape".into();
        assert!(policy.validate().is_err());
    }

    #[test]
    #[ignore = "requires the retained synthetic d003 profile and reviewed policy; CPU only"]
    fn retained_fp32_profile_matches_reviewed_policy_without_running_a_model() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let policy: Policy = serde_json::from_reader(
            File::open(root.join(
                "artifacts/inference_policies/cls_fp32_static_ort1244_windows_v1/policy.json",
            ))
            .unwrap(),
        )
        .unwrap();
        let events: Vec<Value> = serde_json::from_reader(File::open(root.join("artifacts/diagnostics/ort_placement_static_cls_fp32/placement_profile_2026-09-21_05-16-41.json")).unwrap()).unwrap();
        assert_eq!(policy.verify_events(&events).unwrap(), (552, 1304, 276));
    }
}
