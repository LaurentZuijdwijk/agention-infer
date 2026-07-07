use std::collections::HashMap;

use crate::error::GgufError;

/// GGML tensor element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum GgmlType {
    // Discriminants must match ggml.h `enum ggml_type` exactly — they are the
    // on-disk tensor type ids. Note 4 (Q4_2) and 5 (Q4_3) were removed, so the
    // numbering is NOT contiguous here.
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4 = Q4_2 (removed), 5 = Q4_3 (removed)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> std::result::Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            31 => Ok(Self::Q4_0_4_4),
            32 => Ok(Self::Q4_0_4_8),
            33 => Ok(Self::Q4_0_8_8),
            _ => Err(GgufError::UnknownTensorType(v)),
        }
    }

    /// Block size: number of elements per quantization block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::BF16 => 1,
            Self::Q8_0 => 32,
            Self::Q8_1 => 32,
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q5_0 => 32,
            Self::Q5_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
            Self::IQ2_XXS => 256,
            Self::IQ2_XS => 256,
            Self::IQ3_XXS => 256,
            Self::IQ1_S => 256,
            Self::IQ4_NL => 32,
            Self::IQ3_S => 256,
            Self::IQ2_S => 256,
            Self::IQ4_XS => 256,
            Self::IQ1_M => 256,
            Self::I8 => 1,
            Self::I16 => 1,
            Self::I32 => 1,
            Self::I64 => 1,
            Self::F64 => 1,
            Self::Q4_0_4_4 => 32,
            Self::Q4_0_4_8 => 32,
            Self::Q4_0_8_8 => 32,
        }
    }

    /// Byte size of one quantization block.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q8_0 => 34, // 2 (f16 scale) + 32 (i8 values)
            Self::Q8_1 => 40, // 2 (f16 scale) + 2 (f16 min) + 32 (i8 values) + 2 padding
            Self::Q4_0 => 18, // 2 (f16 scale) + 16 (4-bit values)
            Self::Q4_1 => 20, // 2 (f16 scale) + 2 (f16 min) + 16 (4-bit values)
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q2_K => 256,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,
            Self::Q4_0_4_4 => 18,
            Self::Q4_0_4_8 => 18,
            Self::Q4_0_8_8 => 18,
        }
    }

    /// Total bytes needed for `n` elements of this type.
    pub fn type_size(&self, n: usize) -> usize {
        let block_size = self.block_size();
        let block_bytes = self.block_bytes();
        ((n + block_size - 1) / block_size) * block_bytes
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ1_M => "IQ1_M",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Q4_0_4_4 => "Q4_0_4_4",
            Self::Q4_0_4_8 => "Q4_0_4_8",
            Self::Q4_0_8_8 => "Q4_0_8_8",
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArrayType, Vec<MetadataValue>),
    U64(u64),
    I64(i64),
    Float64(f64),
}

/// Element type for metadata arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataArrayType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    Float32,
    Bool,
    String,
    U64,
    I64,
    Float64,
}

impl MetadataArrayType {
    pub fn from_u32(v: u32) -> std::result::Result<Self, GgufError> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::UnknownMetadataType(v)),
        }
    }
}

impl MetadataValue {
    /// Get the GGUF type ID for this value.
    pub fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_, _) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::Float64(_) => 12,
        }
    }

    /// Try to get a u64 from this value, handling U32/U64 coercion.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get an i64 from this value, handling all signed/unsigned integer types.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::U8(v) => Some(*v as i64),
            Self::I8(v) => Some(*v as i64),
            Self::U16(v) => Some(*v as i64),
            Self::I16(v) => Some(*v as i64),
            Self::U32(v) => Some(*v as i64),
            Self::I32(v) => Some(*v as i64),
            Self::U64(v) => Some(*v as i64),
            Self::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get a string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get the elements of an array value.
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(_, items) => Some(items),
            _ => None,
        }
    }

    /// Try to get f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to get bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Format for display.
    pub fn display_value(&self) -> String {
        match self {
            Self::U8(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::Float32(v) => format!("{v:.6}"),
            Self::Bool(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::Array(elem_type, items) => {
                format!("Array({elem_type:?}, {} items)", items.len())
            }
            Self::U64(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::Float64(v) => format!("{v:.6}"),
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub byte_offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Byte size of this tensor's data.
    pub fn byte_size(&self) -> usize {
        self.ggml_type.type_size(self.n_elements())
    }

    /// Shape as a human-readable string, e.g. "[4096, 4096]".
    pub fn shape_str(&self) -> String {
        format!(
            "[{}]",
            self.dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Parsed GGUF file contents.
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where the tensor data section begins (after alignment).
    pub data_offset: u64,
}

impl GgufFile {
    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get a u64 metadata value (handles U32/U64 coercion).
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    /// Get an f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get a bool metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(|v| v.as_bool())
    }

    /// Get a string-array metadata value (e.g. tokenizer.ggml.tokens).
    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        let items = self.metadata.get(key)?.as_array()?;
        items.iter().map(|v| v.as_str()).collect()
    }

    /// Get an integer-array metadata value (e.g. tokenizer.ggml.token_type,
    /// or per-layer lfm2.attention.head_count_kv). Returns None if the key is
    /// absent or not an array of integers.
    pub fn get_i64_array(&self, key: &str) -> Option<Vec<i64>> {
        let items = self.metadata.get(key)?.as_array()?;
        items.iter().map(|v| v.as_i64()).collect()
    }

    /// Get the model architecture string.
    pub fn architecture(&self) -> Option<&str> {
        self.get_string("general.architecture")
    }

    /// Get the model name.
    pub fn name(&self) -> Option<&str> {
        self.get_string("general.name")
    }

    /// Total size of all tensor data in bytes.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_size() as u64).sum()
    }
}
