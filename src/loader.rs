use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::Result;
use crate::parser;
use crate::types::GgufFile;

/// Load and parse a GGUF file using memory mapping.
///
/// The returned `GgufFile` contains parsed metadata and tensor info.
/// The `Mmap` must be kept alive for any tensor data access.
pub fn load(path: &Path) -> Result<(GgufFile, Mmap)> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = parser::parse(&mmap)?;
    Ok((gguf, mmap))
}

/// Parse the split index from a multi-file GGUF filename.
///
/// Returns `(file_index, total_files)` (1-based).
/// Pattern: `name-00001-of-00004.gguf`
pub fn parse_split_index(name: &str) -> Option<(usize, usize)> {
    let name = name.strip_suffix(".gguf")?;
    let of_pos = name.rfind("-of-")?;
    let after_of = &name[of_pos + 4..];
    let before_of = &name[..of_pos];
    let dash_pos = before_of.rfind('-')?;
    let index_str = &before_of[dash_pos + 1..];
    let total_str = after_of;

    let index: usize = index_str.parse().ok()?;
    let total: usize = total_str.parse().ok()?;
    Some((index, total))
}

/// Multi-file GGUF loader for split models.
///
/// Only the first file contains the GGUF header. Subsequent files contain
/// only tensor data with offsets continuing from where the previous file ended.
pub struct MultiFileMmap {
    mmaps: Vec<(Mmap, u64)>, // (mmap, cumulative_start_byte_offset)
}

impl MultiFileMmap {
    /// Load all split files for a multi-file GGUF model.
    ///
    /// `first_file_path` should point to file 1 of N.
    pub fn load(first_file_path: &Path) -> Result<(GgufFile, Self)> {
        let (gguf, first_mmap) = load(first_file_path)?;
        let first_len = first_mmap.len() as u64;

        let name = first_file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let (_, total) = parse_split_index(&name).unwrap_or((1, 1));

        let mut mmaps = vec![(first_mmap, 0u64)];
        let mut cumulative = first_len;

        for i in 2..=total {
            let sibling = sibling_path(first_file_path, i, total)?;
            let file = File::open(&sibling)?;
            let mmap = unsafe { Mmap::map(&file)? };
            mmaps.push((mmap, cumulative));
            cumulative += mmaps.last().unwrap().0.len() as u64;
        }

        Ok((gguf, Self { mmaps }))
    }

    /// Get a byte slice at the given offset and length.
    pub fn slice(&self, offset: u64, len: usize) -> Option<&[u8]> {
        for (mmap, start) in &self.mmaps {
            let end = start + mmap.len() as u64;
            if *start <= offset && offset + len as u64 <= end {
                let local_offset = (offset - *start) as usize;
                return Some(&mmap[local_offset..local_offset + len]);
            }
        }
        None
    }
}

/// Construct the path for the i-th split file by replacing the index in the filename.
fn sibling_path(first: &Path, index: usize, total: usize) -> Result<std::path::PathBuf> {
    let name = first
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let old_pattern = format!("-00001-of-{total:05}");
    let new_pattern = format!("-{index:05}-of-{total:05}");
    let new_name = name.replace(&old_pattern, &new_pattern);
    Ok(first
        .parent()
        .unwrap_or(Path::new("."))
        .join(new_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_split_index() {
        assert_eq!(
            parse_split_index("MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"),
            Some((1, 4))
        );
        assert_eq!(
            parse_split_index("model-00003-of-00005.gguf"),
            Some((3, 5))
        );
        assert_eq!(parse_split_index("single.gguf"), None);
    }
}
