use gguf_rs::load;
use gguf_rs::model::{create_model, KvCache, WeightMap};
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let (gguf, mmap) = load(Path::new(&path)).unwrap();
    let data = &mmap[gguf.data_offset as usize..];
    let cfg = create_model(&gguf, data).unwrap().config().clone();

    // 1. Time a full forward (decode step).
    let mut m = create_model(&gguf, data).unwrap();
    let mut kv = KvCache::new(cfg.block_count as usize, cfg.head_count_kv as usize, cfg.head_dim as usize, 8);
    let _ = m.forward(785, 0, &mut kv).unwrap(); // warm
    let t = Instant::now();
    let _ = m.forward(6722, 1, &mut kv).unwrap();
    println!("full forward:                {:>8.1} ms", t.elapsed().as_secs_f64()*1e3);

    // 2. Time dequantizing every weight tensor ONCE (what we redo each token).
    let wm = WeightMap::from_gguf(&gguf, data);
    let t = Instant::now();
    let mut total = 0usize;
    for tinfo in &gguf.tensors {
        if let Ok(v) = wm.dequant_tensor(&tinfo.name) { total += v.len(); }
    }
    println!("dequant ALL weights once:    {:>8.1} ms  ({} M elems)", t.elapsed().as_secs_f64()*1e3, total/1_000_000);

    // 3. Time just the lm_head / token_embd dequant (tied output projection).
    let t = Instant::now();
    let v = wm.dequant_tensor("token_embd.weight").unwrap();
    println!("dequant token_embd (lm_head):{:>8.1} ms  ({} M elems)", t.elapsed().as_secs_f64()*1e3, v.len()/1_000_000);
}
