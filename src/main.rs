//extern crate ocl;
use ocl::builders::ProgramBuilder;
use ocl::{Device, Platform, ProQue, SpatialDims};
use std::env;
use std::time::SystemTime;

enum Mode {
    SetCnt = 1,
    Trim = 2,
    Extract = 3,
}
const RES_BUFFER_SIZE: usize = 4_000_000;
const LOCAL_WORK_SIZE: u32 = 256;
const GLOBAL_WORK_SIZE: u32 = 1024 * LOCAL_WORK_SIZE;

const TRIMS: u32 = 60;

fn find_paltform(selector: Option<&String>) -> Option<Platform> {
    match selector {
        None => Some(Platform::default()),
        Some(sel) => Platform::list().into_iter().find(|p| {
            if let Ok(vendor) = p.vendor() {
                vendor.contains(sel)
            } else {
                false
            }
        }),
    }
}

fn find_device(platform: &Platform, selector: Option<usize>) -> ocl::Result<Device> {
    match selector {
        None => Device::first(platform),
        Some(index) => Device::by_idx_wrap(platform, index),
    }
}

fn main() -> ocl::Result<()> {
    let args: Vec<String> = env::args().collect();
    let start = SystemTime::now();

    let platform_selector = if args.len() >= 2 {
        Some(&args[1])
    } else {
        None
    };

    let device_selector = if args.len() >= 3 {
        if let Ok(v) = &args[2].parse::<usize>() {
            Some(*v)
        } else {
            return Err("Device ID must be a number".into());
        }
    } else {
        None
    };

    let platform = find_paltform(platform_selector)
        .ok_or::<ocl::Error>("Can't find OpenCL platform".into())?;
    println!("Platform selected: {}", platform.vendor()?);
    let device = find_device(&platform, device_selector)?;
    println!("Device selected: {}", device.to_string());

    let _edge_bits = 29;
    let edge_count = 1024 * 1024 * 512 / 8; //1 << edge_bits;
    let node_count = 1024 * 1024 * 512 / 32; //edge_count * 2;
    let mut res_buf: Vec<u32> = vec![0; RES_BUFFER_SIZE];

    let mut prog_builder = ProgramBuilder::new();
    prog_builder.source_file("./src/lean.cl");

    let m1 = SystemTime::now();
    println!("Preparing {:?}", m1.duration_since(start).unwrap());

    let pro_que = ProQue::builder()
        .prog_bldr(prog_builder)
        .device(&device)
        .dims(GLOBAL_WORK_SIZE)
        .build()?;

    let edges = pro_que
        .buffer_builder::<u8>()
        .len(edge_count)
        .fill_val(0xFF)
        .build()?;
    let counters = pro_que
        .buffer_builder::<u32>()
        .len(node_count)
        .fill_val(0)
        .build()?;

    let result = unsafe {
        pro_que
            .buffer_builder::<u32>()
            .len(RES_BUFFER_SIZE)
            .fill_val(0)
            .use_host_slice(&res_buf[..])
            .build()?
    };

    // random K0 K1 K2 K3 header for testing
    // this is Grin header Blake2 hash represented as 4 x 64bit
    let k0: u64 = 0xa34c6a2bdaa03a14;
    let k1: u64 = 0xd736650ae53eee9e;
    let k2: u64 = 0x9a22f05e3bffed5e;
    let k3: u64 = 0xb8d55478fa3a606d;

    let local_work_size = 256;
    let global_work_size = 1024 * local_work_size;
    let mut current_mode = Mode::SetCnt;
    let mut current_uorv: u32 = 0;

    let mut kernel = pro_que
        .kernel_builder("LeanRound")
        .local_work_size(SpatialDims::One(local_work_size))
        .arg(k0)
        .arg(k1)
        .arg(k2)
        .arg(k3)
        .arg(&edges)
        .arg(&counters)
        .arg(&result)
        .arg(current_mode as u32)
        .arg(current_uorv)
        .build()?;

    // CUCKATOO 29 ONLY at this time, otherwise universal !!!
    //let logsize: u32 = 29; // this needs to be passed to kernel as well AND loops below updated
    //let edges = 1 << logsize;
    let mut offset;

    macro_rules! kernel_enq (
        ($num:expr) => (
        for i in 0..$num {
            offset = i * global_work_size;
            unsafe {
                kernel
                    .set_default_global_work_offset(SpatialDims::One(offset))
                    .enq()?;
            }
        }
        ));

    for l in 0..TRIMS {
        current_uorv = l & 1 as u32;
        current_mode = Mode::SetCnt;
        kernel.set_arg(7, current_mode as u32)?;
        kernel.set_arg(8, current_uorv)?;
        kernel_enq!(8);

        current_mode = if l == (TRIMS - 1) {
            Mode::Extract
        } else {
            Mode::Trim
        };
        kernel.set_arg(7, current_mode as u32)?;
        kernel_enq!(8);
        // prepare for the next round
        if l != TRIMS - 1 {
            counters.cmd().fill(0, None).enq()?;
        }
    }
    unsafe {
        //result.map().enq()?;
        result.read(&mut res_buf).enq()?;
    }
    pro_que.finish()?;
    let m2 = SystemTime::now();
    println!("Trimming {:?}", m2.duration_since(m1).unwrap());

    println!(
        "Done: size  {}  {} {} {}",
        res_buf[0], res_buf[1], res_buf[2], res_buf[3]
    );
    //println!(
    //    "Done 1: size  {}  {} {} {}",
    //    res_buf[4], res_buf[5], res_buf[6], res_buf[7]
    //);
    //println!(
    //    "Done 2: size  {}  {} {} {}",
    //    res_buf[8], res_buf[9], res_buf[10], res_buf[11]
    //);
    let m3 = SystemTime::now();
    let g = Graph::build(&res_buf);
    let m4 = SystemTime::now();
    println!("Building graph {:?}", m4.duration_since(m3).unwrap());
    println!("Nodes {}", g.lists.len());

    let m5 = SystemTime::now();
    let cycle = g.find();
    let m6 = SystemTime::now();
    println!("Searching graph {:?}", m6.duration_since(m5).unwrap());
    if let Some(cycle) = cycle {
        println!("Cycle {}: {:?}", cycle.len(), cycle);
    }
    Ok(())
}

use fnv::{FnvHashMap, FnvHashSet};

struct Search {
    path: Vec<u32>,
    visited: FnvHashSet<u32>,
}

impl Search {
    pub fn new(node_count: usize) -> Search {
        Search {
            path: Vec::with_capacity(node_count),
            visited: FnvHashSet::with_capacity_and_hasher(node_count, Default::default()),
        }
    }

    fn clear(&mut self) {
        self.path.clear();
        //self.visited.clear();
    }

    fn is_cycle(&self, node: u32) -> bool {
        if self.path.contains(&node) {
            let pos = self.path.iter().position(|&v| v == node).unwrap();
            let diff = self.path.len() - pos;
            if diff > 2 {
                println!("Has node {:?} of {}", pos, self.path.len());
            }
        }
        self.path.len() > 4 && self.path[self.path.len() - 5] == node
    }
}

struct Graph {
    lists: FnvHashMap<u32, Vec<u32>>,
    nonces: FnvHashMap<(u32, u32), u32>,
}

impl Graph {
    pub fn build(edges: &[u32]) -> Graph {
        let edge_count = edges[1] as usize;
        let mut g = Graph {
            lists: FnvHashMap::with_capacity_and_hasher(edge_count, Default::default()),
            nonces: FnvHashMap::with_capacity_and_hasher(edge_count, Default::default()),
        };
        const STEP: usize = 4;
        for i in 1..=edge_count {
            let n1 = edges[i * STEP];
            let n2 = edges[i * STEP + 1];
            if n1 == 0 || n2 == 0 {
                println!("ZERO at {}: {} {} ", i, n1, n2);
            }
            let nonce = edges[i * STEP + 2];
            println!("{}-{}-{}", n1, n2, nonce);
            g.lists.entry(n1).or_insert(Vec::new()).push(n2);
            g.lists.entry(n2).or_insert(Vec::new()).push(n1);
            g.nonces.insert((n1, n2), nonce);
        }
        g
    }
    #[inline]
    fn neighbors(&self, node: u32) -> &Vec<u32> {
        // if self.lists[&node].len() > 2 {
        //     println!("Neighbors {} at {}", self.lists[&node].len(), node);
        // }
        &self.lists[&node]
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = &u32> {
        self.lists.keys()
    }

    pub fn find(&self) -> Option<Vec<u32>> {
        let mut search = Search::new(self.lists.len());
        for node in self.nodes() {
            if let Some(c) = self.find_cycle(*node, &mut search) {
                return Some(c);
            }
            search.clear();
        }
        None
    }

    fn find_cycle(&self, current: u32, search: &mut Search) -> Option<Vec<u32>> {
        search.path.push(current);
        //println!("{}> {}", "\t".repeat(search.path.len()), current);
        if search.path.len() > 5 {
            println!("Long path {}", search.path.len());
        }
        search.visited.insert(current);
        for ns in self.neighbors(current) {
            let ns = ns;
            if !search.visited.contains(&ns) {
                if let Some(c) = self.find_cycle(*ns, search) {
                    return Some(c);
                }
            } else {
                if search.is_cycle(*ns) {
                    println!("Found");
                    return Some(search.path.clone());
                }
            }
        }
        search.path.pop();
        //println!("Not found");
        None
    }
}

/*
const BUCKET_BITS: usize = 6;
const BUCKET_SIZE: usize = 1 << BUCKET_BITS;
const BUCKET_MASK: usize = BUCKET_SIZE - 1;

struct BitSet(Vec<u64>);

impl BitSet {
    fn new(size: usize) -> BitSet {
        BitSet(vec![0; (size + 63) / 64])
    }

    fn size(&self) -> usize {
        self.0.len()
    }

    fn set(&mut self, index: usize) {
        let (bucket, bit) = offset(index);
        self.0[bucket] |= bit;
    }

    fn is_set(&self, index: usize) -> bool {
        let (bucket, bit) = offset(index);
        self.0[bucket] & bit != 0
    }
}

fn offset(index: usize) -> (usize, u64) {
    let bucket = index >> BUCKET_BITS;
    let bit = 1 << (index & BUCKET_MASK) as u64;
    println!("index: {} bucket: {} bit: {}", index, bucket, bit);
    (bucket, bit)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_set() {
        let mut s = BitSet::new(1000);
        println!("size: {}", s.size());
        s.set(990);
        assert!(s.is_set(990));
        assert!(!s.is_set(100));
        assert!(!s.is_set(63));
        s.set(63);
        assert!(s.is_set(63));
    }

}
*/
