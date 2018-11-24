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

const TRIMS: u32 = 128;

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
    let edge_count = 1024 * 1024 * 512 / 8;
    let node_count = 1024 * 1024 * 512 / 32;
    let res_buf: Vec<u32> = vec![0; RES_BUFFER_SIZE];

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
    //let k0: u64 = 0xa34c6a2bdaa03a14;
    //let k1: u64 = 0xd736650ae53eee9e;
    //let k2: u64 = 0x9a22f05e3bffed5e;
    //let k3: u64 = 0xb8d55478fa3a606d;

    let k0: u64 = 0x5c0348cfc71b5ce6;
    let k1: u64 = 0xbf4141b92a45e49;
    let k2: u64 = 0x7282d7893f658b88;
    let k3: u64 = 0x61525294db9b617f;

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
        result.map().enq()?;
        //result.read(&mut res_buf).enq()?;
    }
    pro_que.finish()?;
    let m2 = SystemTime::now();
    println!("Trimming {:?}", m2.duration_since(m1).unwrap());
    println!("Trimmed to {} edges", res_buf[1]);

    let m3 = SystemTime::now();
    let g = Graph::build(&res_buf);
    let m4 = SystemTime::now();
    println!("Building graph {:?}", m4.duration_since(m3).unwrap());
    println!("Number of nodes {}", g.node_count());

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

    #[inline]
    pub fn visit(&mut self, node: u32) {
        self.visited.insert(node);
        self.path.push(node);
    }

    #[inline]
    pub fn leave(&mut self) {
        self.path.pop();
    }

    #[inline]
    pub fn is_visited(&self, node: u32) -> bool {
        self.visited.contains(&node)
    }

    #[inline]
    fn clear(&mut self) {
        self.path.clear();
    }

    fn is_cycle(&self, node: u32) -> bool {
        //  TODO remove after tests
        if self.path.contains(&node) {
            let pos = self.path.iter().position(|&v| v == node).unwrap();
            let diff = (self.path.len() - pos) / 2;
            if diff > 1 {
                println!("Found {}-cycle ", diff);
            }
        }
        self.path.len() > 83 && self.path[self.path.len() - 84] == node
    }
}

struct AdjNode {
    value: u32,
    next: Option<usize>,
}

impl AdjNode {
    fn first(value: u32) -> AdjNode {
        AdjNode { value, next: None }
    }

    fn next(value: u32, next: usize) -> AdjNode {
        AdjNode {
            value,
            next: Some(next),
        }
    }
}

struct AdjList<'a> {
    current: Option<&'a AdjNode>,
    adj_store: &'a Vec<AdjNode>,
}

impl<'a> AdjList<'a> {
    pub fn new(current: Option<&'a AdjNode>, adj_store: &'a Vec<AdjNode>) -> AdjList<'a> {
        AdjList { current, adj_store }
    }
}

impl<'a> Iterator for AdjList<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.current {
            None => None,
            Some(node) => {
                let val = node.value;
                match node.next {
                    None => self.current = None,
                    Some(next_index) => self.current = Some(&self.adj_store[next_index]),
                }
                Some(val)
            }
        }
    }
}

struct Graph {
    adj_index: FnvHashMap<u32, usize>,
    adj_store: Vec<AdjNode>,
    nonces: FnvHashMap<(u32, u32), u32>,
}

impl Graph {
    pub fn build(edges: &[u32]) -> Graph {
        let edge_count = edges[1] as usize;
        let mut g = Graph {
            adj_index: FnvHashMap::with_capacity_and_hasher(edge_count * 2, Default::default()),
            nonces: FnvHashMap::with_capacity_and_hasher(edge_count, Default::default()),
            adj_store: Vec::with_capacity(edge_count * 2),
        };
        const STEP: usize = 4;
        for i in 1..=edge_count {
            let n1 = edges[i * STEP];
            let n2 = edges[i * STEP + 1];
            let nonce = edges[i * STEP + 2];
            g.add_edge(n1, n2);
            g.nonces.insert((n1, n2), nonce);
        }
        g
    }

    fn edge_count(&self) -> usize {
        self.nonces.len()
    }

    fn node_count(&self) -> usize {
        self.adj_index.len()
    }

    fn add_edge(&mut self, node1: u32, node2: u32) {
        self.add_half_edge(node1, node2);
        self.add_half_edge(node2, node1);
    }

    fn add_half_edge(&mut self, from: u32, to: u32) {
        if let Some(index) = self.adj_index.get_mut(&from) {
            self.adj_store.push(AdjNode::next(to, *index));
            *index = self.adj_store.len() - 1;
        } else {
            self.adj_store.push(AdjNode::first(to));
            self.adj_index.insert(from, self.adj_store.len() - 1);
        }
    }

    #[inline]
    fn neighbors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let node = match self.adj_index.get(&node) {
            Some(index) => Some(&self.adj_store[*index]),
            None => None,
        };
        AdjList::new(node, &self.adj_store)
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = &u32> {
        self.adj_index.keys()
    }

    pub fn find(&self) -> Option<Vec<u32>> {
        let mut search = Search::new(self.node_count());
        for node in self.nodes() {
            if let Some(c) = self.walk_graph(*node, &mut search) {
                return Some(c);
            }
            search.clear();
        }
        None
    }

    fn walk_graph(&self, current: u32, search: &mut Search) -> Option<Vec<u32>> {
        if search.is_visited(current) {
            return None;
        }
        search.visit(current);
        for ns in self.neighbors(current) {
            if !search.is_visited(ns) {
                search.visit(ns);
                if let Some(c) = self.walk_graph(ns ^ 1, search) {
                    return Some(c);
                }
                search.leave();
            } else {
                if search.is_cycle(ns) {
                    println!("Found");
                    return Some(search.path.clone());
                }
            }
        }
        search.leave();
        None
    }
}
