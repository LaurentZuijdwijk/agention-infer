use thiserror::Error;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("invalid GGUF magic bytes")]
    InvalidMagic,

    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("unexpected end of input at byte {0}")]
    UnexpectedEof(usize),

    #[error("unknown metadata value type: {0}")]
    UnknownMetadataType(u32),

    #[error("unknown tensor GGML type: {0}")]
    UnknownTensorType(u32),

    #[error("invalid string at byte offset {0}")]
    InvalidString(usize),

    #[error("missing required metadata key: {0}")]
    MissingMetadata(String),

    #[error("wrong type for metadata key {0}: expected {1}")]
    WrongMetadataType(String, &'static str),

    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("multi-file mismatch: {0}")]
    MultiFileMismatch(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("backend error: {0}")]
    BackendError(String),
}

pub type Result<T> = std::result::Result<T, GgufError>;
