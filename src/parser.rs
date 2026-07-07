use std::collections::HashMap;

use crate::error::{GgufError, Result};
use crate::types::{GgmlType, GgufFile, MetadataArrayType, MetadataValue, TensorInfo};

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF" in little-endian

/// Parse a GGUF file from raw bytes. Pure function — no I/O.
pub fn parse(data: &[u8]) -> Result<GgufFile> {
    let mut cursor = Cursor::new(data);

    // Header
    let magic = cursor.read_bytes(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }

    let version = cursor.read_u32()?;
    if version < 2 || version > 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = cursor.read_u64()?;
    let metadata_kv_count = cursor.read_u64()?;

    // Metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = cursor.read_string()?;
        let value_type_id = cursor.read_u32()?;
        let value = cursor.read_metadata_value(value_type_id)?;
        metadata.insert(key, value);
    }

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = cursor.read_string()?;
        let n_dims = cursor.read_u32()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(cursor.read_u64()?);
        }
        let ggml_type_id = cursor.read_u32()?;
        let ggml_type = GgmlType::from_u32(ggml_type_id)?;
        let byte_offset = cursor.read_u64()?;
        tensors.push(TensorInfo {
            name,
            n_dims,
            dims,
            ggml_type,
            byte_offset,
        });
    }

    // Data section starts at next 32-byte aligned position after cursor
    let data_offset = align_to(cursor.pos, 32) as u64;

    Ok(GgufFile {
        version,
        tensor_count,
        metadata,
        tensors,
        data_offset,
    })
}

fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof(self.pos));
        }
        let result = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(result)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> Result<bool> {
        let v = self.read_u8()?;
        Ok(v != 0)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len {
            return Err(GgufError::UnexpectedEof(self.pos));
        }
        let bytes = &self.data[self.pos..self.pos + len];
        self.pos += len;
        let s = String::from_utf8(bytes.to_vec())
            .map_err(|_| GgufError::InvalidString(self.pos - len))?;
        Ok(s)
    }

    fn read_metadata_value(&mut self, type_id: u32) -> Result<MetadataValue> {
        match type_id {
            0 => Ok(MetadataValue::U8(self.read_u8()?)),
            1 => Ok(MetadataValue::I8(self.read_i8()?)),
            2 => Ok(MetadataValue::U16(self.read_u16()?)),
            3 => Ok(MetadataValue::I16(self.read_i16()?)),
            4 => Ok(MetadataValue::U32(self.read_u32()?)),
            5 => Ok(MetadataValue::I32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_bool()?)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                // Array: elem_type (u32) + count (u64) + values
                let elem_type_id = self.read_u32()?;
                let elem_type = MetadataArrayType::from_u32(elem_type_id)?;
                let count = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_metadata_value(elem_type_id)?);
                }
                Ok(MetadataValue::Array(elem_type, items))
            }
            10 => Ok(MetadataValue::U64(self.read_u64()?)),
            11 => Ok(MetadataValue::I64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(GgufError::UnknownMetadataType(type_id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gguf_header(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&metadata_kv_count.to_le_bytes());
        buf
    }

    fn build_string(s: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(bytes);
        buf
    }

    #[test]
    fn test_parse_minimal_v2() {
        let data = build_gguf_header(2, 0, 0);
        // No metadata, no tensors → data section starts at aligned position
        let parsed = parse(&data).unwrap();
        assert_eq!(parsed.version, 2);
        assert_eq!(parsed.tensor_count, 0);
        assert!(parsed.metadata.is_empty());
        assert!(parsed.tensors.is_empty());
    }

    #[test]
    fn test_parse_invalid_magic() {
        let data = [0u8; 24];
        let result = parse(&data);
        assert!(matches!(result, Err(GgufError::InvalidMagic)));
    }

    #[test]
    fn test_parse_unsupported_version() {
        let data = build_gguf_header(1, 0, 0);
        // Overwrite version
        let result = parse(&data);
        assert!(matches!(result, Err(GgufError::UnsupportedVersion(1))));
    }

    #[test]
    fn test_parse_v3_with_metadata() {
        let mut data = build_gguf_header(3, 0, 1);
        // One metadata entry: key="general.architecture", type=STRING(8), value="llama"
        data.extend(build_string("general.architecture"));
        data.extend_from_slice(&8u32.to_le_bytes()); // STRING type
        data.extend(build_string("llama"));

        let parsed = parse(&data).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.get_string("general.architecture"), Some("llama"));
    }

    #[test]
    fn test_parse_metadata_u32() {
        let mut data = build_gguf_header(3, 0, 1);
        data.extend(build_string("test.value"));
        data.extend_from_slice(&4u32.to_le_bytes()); // U32 type
        data.extend_from_slice(&42u32.to_le_bytes());

        let parsed = parse(&data).unwrap();
        assert_eq!(parsed.get_u64("test.value"), Some(42));
    }

    #[test]
    fn test_parse_with_tensor() {
        let mut data = build_gguf_header(3, 1, 0);
        // One tensor: name="output.weight", n_dims=2, dims=[4096,4096], type=Q8_0(8), offset=0
        data.extend(build_string("output.weight"));
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&4096u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&4096u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&8u32.to_le_bytes()); // Q8_0 (ggml type id 8)
        data.extend_from_slice(&0u64.to_le_bytes()); // byte_offset

        let parsed = parse(&data).unwrap();
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(parsed.tensors[0].name, "output.weight");
        assert_eq!(parsed.tensors[0].dims, vec![4096, 4096]);
        assert_eq!(parsed.tensors[0].ggml_type, GgmlType::Q8_0);
    }

    #[test]
    fn test_parse_array_metadata() {
        let mut data = build_gguf_header(3, 0, 1);
        data.extend(build_string("tokenizer.ggml.bos_token_id"));
        // Array of U32 with 1 element
        data.extend_from_slice(&9u32.to_le_bytes()); // ARRAY type
        data.extend_from_slice(&4u32.to_le_bytes()); // elem type: U32
        data.extend_from_slice(&1u64.to_le_bytes()); // count
        data.extend_from_slice(&1u32.to_le_bytes()); // value

        let parsed = parse(&data).unwrap();
        let val = parsed.get("tokenizer.ggml.bos_token_id").unwrap();
        match val {
            MetadataValue::Array(MetadataArrayType::U32, items) => {
                assert_eq!(items.len(), 1);
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_unexpected_eof() {
        // Header only, but metadata_kv_count = 1
        let data = build_gguf_header(3, 0, 1);
        // Don't add any metadata → should get UnexpectedEof
        let result = parse(&data);
        assert!(matches!(result, Err(GgufError::UnexpectedEof(_))));
    }
}
