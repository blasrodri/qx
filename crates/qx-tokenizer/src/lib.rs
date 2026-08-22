//! ByteLevel BPE tokenizer compatible with the Granite checkpoint tokenizer.

use anyhow::{Context, Result};
use regex::Regex;
use std::{collections::HashMap, fs, path::Path, sync::OnceLock};

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
    merges: HashMap<(String, String), usize>,
    special_tokens: HashMap<String, u32>,
    inverse_special_tokens: HashMap<u32, String>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref();
        let vocab_path = root.join("vocab.json");
        let merges_path = root.join("merges.txt");
        let added_tokens_path = root.join("added_tokens.json");
        let vocab = serde_json::from_slice::<HashMap<String, u32>>(
            &fs::read(&vocab_path).with_context(|| format!("read {}", vocab_path.display()))?,
        )
        .with_context(|| format!("parse {}", vocab_path.display()))?;
        let merges = parse_merges(&merges_path)?;
        let special_tokens = if added_tokens_path.exists() {
            serde_json::from_slice::<HashMap<String, u32>>(&fs::read(&added_tokens_path)?)?
        } else {
            HashMap::new()
        };
        Self::new(vocab, merges, special_tokens)
    }

    pub fn new(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        special_tokens: HashMap<String, u32>,
    ) -> Result<Self> {
        let inverse_vocab = vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        let merge_ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect();
        let inverse_special_tokens = special_tokens
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect();
        let (byte_encoder, byte_decoder) = byte_maps();
        Ok(Self {
            vocab,
            inverse_vocab,
            merges: merge_ranks,
            special_tokens,
            inverse_special_tokens,
            byte_encoder,
            byte_decoder,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        for (is_special, segment) in split_special(text, &self.special_tokens) {
            if is_special {
                tokens.push(
                    *self
                        .special_tokens
                        .get(segment)
                        .context("special token disappeared during split")?,
                );
            } else {
                self.encode_regular(segment, &mut tokens)?;
            }
        }
        Ok(tokens)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        let mut output = String::new();
        for token in tokens {
            if let Some(special) = self.inverse_special_tokens.get(token) {
                output.push_str(&decode_bytes(&mut bytes)?);
                output.push_str(special);
                continue;
            }
            let value = self
                .inverse_vocab
                .get(token)
                .with_context(|| format!("token id {token} is not in vocabulary"))?;
            for character in value.chars() {
                let byte = self
                    .byte_decoder
                    .get(&character)
                    .with_context(|| format!("token contains unknown byte symbol {character:?}"))?;
                bytes.push(*byte);
            }
        }
        output.push_str(&decode_bytes(&mut bytes)?);
        Ok(output)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.vocab.get("<|end_of_text|>").copied()
    }

    fn encode_regular(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        static PATTERN: OnceLock<Regex> = OnceLock::new();
        let pattern = PATTERN.get_or_init(|| {
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .expect("valid ByteLevel BPE regex")
        });
        for part in pattern.find_iter(text).map(|match_| match_.as_str()) {
            let symbols = part
                .as_bytes()
                .iter()
                .map(|byte| self.byte_encoder[*byte as usize].to_string())
                .collect::<Vec<_>>();
            for symbol in bpe(symbols, &self.merges) {
                output.push(
                    *self
                        .vocab
                        .get(&symbol)
                        .with_context(|| format!("BPE symbol {symbol:?} is not in vocabulary"))?,
                );
            }
        }
        Ok(())
    }
}

fn parse_merges(path: &Path) -> Result<Vec<(String, String)>> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    contents
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with("#"))
        .map(|line| {
            let mut parts = line.splitn(2, ' ');
            let left = parts.next().unwrap().to_owned();
            let right = parts
                .next()
                .with_context(|| format!("invalid BPE merge {line:?}"))?
                .to_owned();
            Ok((left, right))
        })
        .collect()
}

fn split_special<'a>(text: &'a str, special: &HashMap<String, u32>) -> Vec<(bool, &'a str)> {
    if special.is_empty() {
        return vec![(false, text)];
    }
    let mut names = special.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut segments = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let next = names
            .iter()
            .filter_map(|name| text[cursor..].starts_with(name).then_some(*name))
            .next();
        if let Some(name) = next {
            segments.push((true, &text[cursor..cursor + name.len()]));
            cursor += name.len();
            continue;
        }
        let start = cursor;
        cursor += text[cursor..].chars().next().unwrap().len_utf8();
        while cursor < text.len() && !names.iter().any(|name| text[cursor..].starts_with(name)) {
            cursor += text[cursor..].chars().next().unwrap().len_utf8();
        }
        segments.push((false, &text[start..cursor]));
    }
    segments
}

fn bpe(mut symbols: Vec<String>, merges: &HashMap<(String, String), usize>) -> Vec<String> {
    loop {
        let best = symbols
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| {
                merges
                    .get(&(pair[0].clone(), pair[1].clone()))
                    .map(|rank| (*rank, index))
            })
            .min();
        let Some((_, target)) = best else { break };
        let left = symbols[target].clone();
        let right = symbols[target + 1].clone();
        let mut merged = Vec::with_capacity(symbols.len());
        let mut index = 0;
        while index < symbols.len() {
            if index + 1 < symbols.len() && symbols[index] == left && symbols[index + 1] == right {
                merged.push(format!("{}{}", left, right));
                index += 2;
            } else {
                merged.push(symbols[index].clone());
                index += 1;
            }
        }
        symbols = merged;
    }
    symbols
}

fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bytes = Vec::new();
    for value in 33..=126 {
        bytes.push(value);
    }
    for value in 161..=172 {
        bytes.push(value);
    }
    for value in 174..=255 {
        bytes.push(value);
    }
    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    for (index, byte) in (0_u16..=255).enumerate() {
        let character = if bytes.contains(&(byte as u8)) {
            char::from_u32(byte as u32).unwrap()
        } else {
            let offset = (0..byte)
                .filter(|candidate| !bytes.contains(&(*candidate as u8)))
                .count();
            char::from_u32(256 + offset as u32).unwrap()
        };
        encoder[index] = character;
        decoder.insert(character, byte as u8);
    }
    (encoder, decoder)
}

fn decode_bytes(bytes: &mut Vec<u8>) -> Result<String> {
    let value =
        String::from_utf8(std::mem::take(bytes)).context("decoded BPE bytes are not UTF-8")?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn granite_fixture_matches_reference_token_ids() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../external/models/granite-3.0-1b-a400m-instruct");
        if !path.exists() {
            return;
        }
        let tokenizer = Tokenizer::from_dir(path).unwrap();
        assert_eq!(tokenizer.encode("Hello world").unwrap(), [8279, 5788]);
        assert_eq!(
            tokenizer.encode("Hello, world!").unwrap(),
            [8279, 30, 5788, 19]
        );
        let role = "<|start_of_role|>user<|end_of_role|>\nWhat is 2+2?";
        assert_eq!(
            tokenizer.encode(role).unwrap(),
            [49152, 496, 49153, 203, 8197, 438, 225, 36, 29, 36, 49]
        );
        assert_eq!(tokenizer.decode(&[8279, 5788]).unwrap(), "Hello world");
    }
}
