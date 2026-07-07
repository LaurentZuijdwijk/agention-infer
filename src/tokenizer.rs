//! GPT-2 byte-level BPE tokenizer, loaded from GGUF metadata.
//!
//! LFM2 (and Llama-3, Qwen2, etc.) ship a `tokenizer.ggml.model = gpt2`
//! tokenizer: byte-level BPE with a merge table. There is no byte-fallback —
//! every one of the 256 raw bytes is mapped to a printable Unicode char via
//! the GPT-2 `bytes_to_unicode` table, so any input is representable.
//!
//! Pipeline:
//!   text ──pretokenize──▶ words ──byte-level encode──▶ symbol strings
//!        ──BPE merges──▶ pieces ──vocab lookup──▶ token ids
//!
//! Special/control tokens (`<|im_start|>`, `<|im_end|>`, …) are matched
//! literally *before* BPE so they map to their single reserved id.

use std::collections::HashMap;

use crate::error::{GgufError, Result};
use crate::types::GgufFile;

/// GGUF token type ids (llama.cpp `llama_token_type`).
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;

/// A byte-level BPE tokenizer.
pub struct Tokenizer {
    /// token string (byte-level unicode form) → id
    vocab: HashMap<String, u32>,
    /// id → token string (byte-level unicode form)
    tokens: Vec<String>,
    /// merge rank: lower rank = higher priority. Key is the concatenation
    /// `left + "\u{1}" + right` (the U+0001 separator can't occur in a
    /// byte-level token string, which only uses the GPT-2 printable set).
    merge_ranks: HashMap<String, u32>,
    /// byte value → mapped unicode char
    byte_to_char: [char; 256],
    /// mapped unicode char → byte value
    char_to_byte: HashMap<char, u8>,
    /// special tokens (control + user-defined), longest first for greedy match
    special: Vec<(String, u32)>,

    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

impl Tokenizer {
    /// Build a tokenizer from parsed GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let model = gguf
            .get_string("tokenizer.ggml.model")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.model".into()))?;
        if model != "gpt2" {
            return Err(GgufError::BackendError(format!(
                "unsupported tokenizer model {model:?} (only gpt2 BPE is implemented)"
            )));
        }

        let tokens: Vec<String> = gguf
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".into()))?
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let token_types = gguf
            .get_i64_array("tokenizer.ggml.token_type")
            .unwrap_or_default();

        let merges = gguf
            .get_str_array("tokenizer.ggml.merges")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.merges".into()))?;

        let mut vocab = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), id as u32);
        }

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // Each merge is "left right" separated by a single ASCII space.
            if let Some((l, r)) = m.split_once(' ') {
                merge_ranks.insert(format!("{l}\u{1}{r}"), rank as u32);
            }
        }

        let (byte_to_char, char_to_byte) = build_byte_maps();

        // Collect special tokens (control + user-defined). Sort longest-first so
        // greedy scanning prefers the most specific match.
        let mut special: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| {
                matches!(
                    token_types.get(*id).copied(),
                    Some(TOKEN_TYPE_CONTROL) | Some(TOKEN_TYPE_USER_DEFINED)
                )
            })
            .map(|(id, tok)| (tok.clone(), id as u32))
            .collect();
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let bos_token_id = gguf
            .get_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as u32);
        let eos_token_id = gguf
            .get_u64("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32);
        let add_bos = gguf.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = gguf.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);

        Ok(Self {
            vocab,
            tokens,
            merge_ranks,
            byte_to_char,
            char_to_byte,
            special,
            bos_token_id,
            eos_token_id,
            add_bos,
            add_eos,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Look up the string form (decoded UTF-8, may be lossy) of a single id.
    pub fn id_to_piece(&self, id: u32) -> String {
        let raw = match self.tokens.get(id as usize) {
            Some(s) => s,
            None => return String::new(),
        };
        let bytes = self.piece_to_bytes(raw);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Raw bytes a single token decodes to. Use this for streaming: accumulate
    /// bytes across tokens and flush complete UTF-8, since a multi-byte char may
    /// span two tokens.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.tokens
            .get(id as usize)
            .map(|raw| self.piece_to_bytes(raw))
            .unwrap_or_default()
    }

    /// Encode text into token ids, honoring `add_bos`/`add_eos`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        if self.add_bos {
            if let Some(bos) = self.bos_token_id {
                ids.push(bos);
            }
        }
        self.encode_into(text, &mut ids);
        if self.add_eos {
            if let Some(eos) = self.eos_token_id {
                ids.push(eos);
            }
        }
        ids
    }

    /// Encode without adding BOS/EOS (used for chat templates that manage
    /// their own control tokens).
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        self.encode_into(text, &mut ids);
        ids
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        // Split around special tokens first; encode the gaps with BPE.
        let mut rest = text;
        while !rest.is_empty() {
            match self.find_next_special(rest) {
                Some((start, len, id)) => {
                    if start > 0 {
                        self.bpe_encode(&rest[..start], out);
                    }
                    out.push(id);
                    rest = &rest[start + len..];
                }
                None => {
                    self.bpe_encode(rest, out);
                    break;
                }
            }
        }
    }

    /// Find the earliest occurrence of any special token in `text`.
    /// Returns (byte_start, byte_len, id).
    fn find_next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (tok, id) in &self.special {
            if let Some(pos) = text.find(tok.as_str()) {
                let better = match best {
                    None => true,
                    // earlier position wins; on a tie the longer match wins
                    Some((bpos, blen, _)) => pos < bpos || (pos == bpos && tok.len() > blen),
                };
                if better {
                    best = Some((pos, tok.len(), *id));
                }
            }
        }
        best
    }

    /// Byte-level BPE over a plain-text span (no special tokens inside).
    fn bpe_encode(&self, text: &str, out: &mut Vec<u32>) {
        for word in pretokenize(text) {
            // Map the word's raw UTF-8 bytes to the byte-level unicode symbols.
            let mut symbols: Vec<String> = word
                .bytes()
                .map(|b| self.byte_to_char[b as usize].to_string())
                .collect();
            if symbols.is_empty() {
                continue;
            }

            self.merge_symbols(&mut symbols);

            for sym in &symbols {
                match self.vocab.get(sym) {
                    Some(&id) => out.push(id),
                    None => {
                        // Should not happen with a complete byte-level vocab,
                        // but fall back to per-symbol lookup to stay lossless.
                        for ch in sym.chars() {
                            let s = ch.to_string();
                            if let Some(&id) = self.vocab.get(&s) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Repeatedly merge the highest-priority adjacent pair until none remain.
    fn merge_symbols(&self, symbols: &mut Vec<String>) {
        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = format!("{}\u{1}{}", symbols[i], symbols[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }
            let Some(i) = best_idx else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
    }

    /// Decode a slice of ids back into a UTF-8 string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(raw) = self.tokens.get(id as usize) {
                bytes.extend_from_slice(&self.piece_to_bytes(raw));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Map a byte-level unicode token string back to its raw bytes.
    fn piece_to_bytes(&self, raw: &str) -> Vec<u8> {
        raw.chars()
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect()
    }
}

/// Build the GPT-2 `bytes_to_unicode` bijection.
fn build_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    // Bytes that map to themselves (printable ranges).
    let mut direct = [false; 256];
    for b in b'!'..=b'~' {
        direct[b as usize] = true;
    }
    for b in 0xA1u16..=0xAC {
        direct[b as usize] = true;
    }
    for b in 0xAEu16..=0xFF {
        direct[b as usize] = true;
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut next = 256u32; // codepoints for the non-printable bytes

    for b in 0..256usize {
        let cp = if direct[b] {
            b as u32
        } else {
            let c = next;
            next += 1;
            c
        };
        let ch = char::from_u32(cp).expect("valid codepoint");
        byte_to_char[b] = ch;
        char_to_byte.insert(ch, b as u8);
    }

    (byte_to_char, char_to_byte)
}

/// GPT-2 pretokenization, implemented manually (Rust's `regex` has no
/// lookahead, and the GPT-2 pattern needs `\s+(?!\S)`).
///
/// Pattern approximated:
///   's|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ |
///   \s+(?!\S) | \s+
///
/// Returns the list of word spans (as owned strings) in order.
fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut words = Vec::new();
    let mut i = 0;

    while i < n {
        // Contractions: '  followed by known suffix.
        if chars[i] == '\'' && i + 1 < n {
            if let Some(len) = match_contraction(&chars[i..]) {
                words.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // Optional single leading space, then a run of one category.
        let has_space = chars[i] == ' ';
        let cat_start = if has_space { i + 1 } else { i };

        if cat_start < n {
            let c = chars[cat_start];
            if is_letter(c) {
                let mut j = cat_start + 1;
                while j < n && is_letter(chars[j]) {
                    j += 1;
                }
                words.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            if is_number(c) {
                let mut j = cat_start + 1;
                while j < n && is_number(chars[j]) {
                    j += 1;
                }
                words.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            if !c.is_whitespace() {
                // ` ?[^\s\p{L}\p{N}]+`
                let mut j = cat_start + 1;
                while j < n && !chars[j].is_whitespace() && !is_letter(chars[j]) && !is_number(chars[j])
                {
                    j += 1;
                }
                words.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }

        // Whitespace run. `\s+(?!\S)` means: a run of whitespace, but if it is
        // followed by a non-space, leave the last space to prefix that token.
        let mut j = i;
        while j < n && chars[j].is_whitespace() {
            j += 1;
        }
        let mut end = j;
        if j < n && end - i > 1 {
            // trailing space belongs to the following word
            end -= 1;
        }
        if end > i {
            words.push(chars[i..end].iter().collect());
        }
        i = end;
    }

    words
}

/// Match a GPT-2 contraction suffix at the start of `chars` (which begins with `'`).
/// Returns the length in chars including the apostrophe.
fn match_contraction(chars: &[char]) -> Option<usize> {
    let s: String = chars.iter().take(4).collect();
    for suf in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if s.starts_with(suf) {
            return Some(suf.chars().count());
        }
    }
    None
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_maps_are_bijective() {
        let (b2c, c2b) = build_byte_maps();
        for b in 0..256usize {
            assert_eq!(c2b[&b2c[b]], b as u8);
        }
        assert_eq!(c2b.len(), 256);
        // Known GPT-2 anchors.
        assert_eq!(b2c[b' ' as usize], '\u{0120}'); // 'Ġ'
        assert_eq!(b2c[b'\n' as usize], '\u{010A}'); // 'Ċ'
        assert_eq!(b2c[b'!' as usize], '!');
    }

    #[test]
    fn pretokenize_basic() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize("it's"), vec!["it", "'s"]);
        assert_eq!(pretokenize("a1"), vec!["a", "1"]);
    }

    #[test]
    fn pretokenize_trailing_space_prefixes_next() {
        // Two spaces: first is a standalone ws run, second prefixes "b".
        assert_eq!(pretokenize("a  b"), vec!["a", " ", " b"]);
    }
}
