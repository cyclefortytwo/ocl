extern crate int_hash;
extern crate ocl;

use ocl::flags::CommandQueueProperties;
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue, SpatialDims};
use std::env;
use std::time::SystemTime;

enum Mode {
    SetCnt = 1,
    Trim = 2,
    Extract = 3,
}
const RES_BUFFER_SIZE: usize = 4_000_000;
const LOCAL_WORK_SIZE: usize = 256;
const GLOBAL_WORK_SIZE: usize = 1024 * LOCAL_WORK_SIZE;

const TRIMS: u32 = 256;

fn find_paltform(selector: Option<&String>) -> Option<Platform> {
    match selector {
        None => Some(Platform::default()),
        Some(sel) => Platform::list().into_iter().find(|p| {
            if let Ok(vendor) = p.name() {
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

    let _edge_bits = 31;
    let edge_count = 1024 * 1024 * 64;
    let node_count = 1024 * 1024 * 64;
    let res_buf: Vec<u32> = vec![0; RES_BUFFER_SIZE];

    let m1 = SystemTime::now();
    println!("Preparing {:?}", m1.duration_since(start).unwrap());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let qu = Queue::new(&context, device, Some(CommandQueueProperties::new()))?;

    let prog = Program::builder()
        .devices(device)
        .source_file("./src/lean.cl")
        .build(&context)?;

    let edges = Buffer::<u32>::builder()
        .queue(qu.clone())
        .len(edge_count)
        .fill_val(0xFFFFFFFF)
        .build()?;
    let counters = Buffer::<u32>::builder()
        .queue(qu.clone())
        .len(node_count)
        .fill_val(0)
        .build()?;
    let result = unsafe {
        Buffer::<u32>::builder()
            .queue(qu.clone())
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

    //let k0: u64 = 0x5c0348cfc71b5ce6;
    //let k1: u64 = 0xbf4141b92a45e49;
    //let k2: u64 = 0x7282d7893f658b88;
    //let k3: u64 = 0x61525294db9b617f;

    //let k0: u64 = 0x27580576fe290177;
    //let k1: u64 = 0xf9ea9b2031f4e76e;
    //let k2: u64 = 0x1663308c8607868f;
    //let k3: u64 = 0xb88839b0fa180d0e;
	
	// cuckatoo31
	//8785f61f3e087286 91b57e6072a0cdaa 8035f9ee251a77a0 de03da786148f07
	let k0: u64 = 0x8785f61f3e087286;
    let k1: u64 = 0x91b57e6072a0cdaa;
    let k2: u64 = 0x8035f9ee251a77a0;
    let k3: u64 = 0xde03da786148f07;

    let mut current_mode = Mode::SetCnt;
    let mut current_uorv: u32 = 0;

    let mut kernel = Kernel::builder()
        .name("LeanRound")
        .program(&prog)
        .queue(qu.clone())
        .global_work_size(GLOBAL_WORK_SIZE)
        .local_work_size(SpatialDims::One(LOCAL_WORK_SIZE))
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
            offset = i * GLOBAL_WORK_SIZE;
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
        kernel_enq!(32);

        current_mode = if l == (TRIMS - 1) {
            Mode::Extract
        } else {
            Mode::Trim
        };
        kernel.set_arg(7, current_mode as u32)?;
        kernel_enq!(32);
        // prepare for the next round
        if l != TRIMS - 1 {
            counters.cmd().fill(0, None).enq()?;
        }
    }
    unsafe {
        result.map().enq()?;
        //result.read(&mut res_buf).enq()?;
    }
    qu.finish()?;
    let m2 = SystemTime::now();
    println!("Trimming {:?}", m2.duration_since(m1).unwrap());
    println!("Trimmed to {} edges", res_buf[1]);

    let m3 = SystemTime::now();
    let g = Graph::build(&res_buf);
    let m4 = SystemTime::now();
    println!("Building graph {:?}", m4.duration_since(m3).unwrap());
    println!(
        "Number of nodes {}, edges {}",
        g.node_count(),
        g.edge_count()
    );

    let m5 = SystemTime::now();
    let _cycle = g.find()?;
    let m6 = SystemTime::now();
    println!("Searching graph {:?}", m6.duration_since(m5).unwrap());
    Ok(())
}

//use fnv::IntHashMap;
use int_hash::IntHashMap;

struct Solution {
    nonces: Vec<u32>,
}

struct Search {
    length: usize,
    path: Vec<u32>,
    solutions: Vec<Solution>,

    state: IntHashMap<u32, NodeState>,
    node_visited: usize,
    node_explored: usize,
}

#[derive(Clone, Copy)]
enum NodeState {
    NotVisited,
    Visited,
    Explored,
}

impl Search {
    pub fn new(node_count: usize, length: usize) -> Search {
        Search {
            path: Vec::with_capacity(node_count),
            solutions: vec![],
            length: length * 2,
            state: IntHashMap::with_capacity_and_hasher(node_count, Default::default()),
            node_visited: 0,
            node_explored: 0,
        }
    }

    #[inline]
    pub fn visit(&mut self, node: u32) {
        self.state.insert(node, NodeState::Visited);
        self.path.push(node);
        self.node_visited += 1;
    }

    #[inline]
    pub fn explore(&mut self, node: u32) {
        self.state.insert(node, NodeState::Explored);
        self.path.push(node);
        self.node_explored += 1;
    }

    #[inline]
    pub fn leave(&mut self) {
        self.path.pop();
    }

    #[inline]
    pub fn state(&self, node: u32) -> NodeState {
        match self.state.get(&node) {
            None => NodeState::NotVisited,
            Some(state) => *state,
        }
    }

    #[inline]
    pub fn is_visited(&self, node: u32) -> bool {
        match self.state(node) {
            NodeState::NotVisited => false,
            _ => true,
        }
    }

    #[inline]
    pub fn is_explored(&self, node: u32) -> bool {
        match self.state(node) {
            NodeState::Explored => true,
            _ => false,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.path.clear();
        //self.state.clear();
    }

    fn is_cycle(&mut self, node: u32) -> bool {
        //  TODO remove after tests
        if self.path.contains(&node) {
            let pos = self.path.iter().position(|&v| v == node).unwrap();
            let diff = (self.path.len() - pos) / 2;
            if diff > 1 {
                println!("Found {}-cycle {}", diff, node);
            }
        }
        let res =
            self.path.len() > self.length - 1 && self.path[self.path.len() - self.length] == node;
        if res {
            self.path.push(node);
        }
        res
    }
}

struct AdjNode {
    value: u32,
    next: Option<usize>,
}

impl AdjNode {
    #[inline]
    fn first(value: u32) -> AdjNode {
        AdjNode { value, next: None }
    }

    #[inline]
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
    #[inline]
    pub fn new(current: Option<&'a AdjNode>, adj_store: &'a Vec<AdjNode>) -> AdjList<'a> {
        AdjList { current, adj_store }
    }
}

impl<'a> Iterator for AdjList<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match self.current {
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

fn nonce_key(node1: u32, node2: u32) -> (u32, u32) {
    if node1 < node2 {
        (node1, node2)
    } else {
        (node2, node1)
    }
}

struct Graph {
    adj_index: IntHashMap<u32, usize>,
    adj_store: Vec<AdjNode>,
    nonces: IntHashMap<(u32, u32), u32>,
}

impl Graph {
    pub fn build(edges: &[u32]) -> Graph {
        let edge_count = edges[1] as usize;
        let mut g = Graph {
            adj_index: IntHashMap::with_capacity_and_hasher(edge_count * 2, Default::default()),
            nonces: IntHashMap::with_capacity_and_hasher(edge_count, Default::default()),
            adj_store: Vec::with_capacity(edge_count * 2),
        };
        const STEP: usize = 4;
        for i in 1..=edge_count {
            let n1 = edges[i * STEP];
            let n2 = edges[i * STEP + 1];
            let nonce = edges[i * STEP + 2];
            g.add_edge(n1, n2);
            g.nonces.insert(nonce_key(n1, n2), nonce);
        }
        g
    }

    pub fn get_nonce(&self, node1: u32, node2: u32) -> Result<u32, String> {
        match self.nonces.get(&nonce_key(node1, node2)) {
            None => Err(format!("can not find  a nonce for {}:{}", node1, node2)),
            Some(v) => Ok(*v),
        }
    }

    #[inline]
    fn node_count(&self) -> usize {
        self.adj_index.len()
    }

    #[inline]
    pub fn edge_count(&self) -> usize {
        self.adj_store.len() / 2
    }

    #[inline]
    fn add_edge(&mut self, node1: u32, node2: u32) {
        self.add_half_edge(node1, node2);
        self.add_half_edge(node2, node1);
    }

    fn add_half_edge(&mut self, from: u32, to: u32) {
        if let Some(index) = self.adj_index.get(&from) {
            self.adj_store.push(AdjNode::next(to, *index));
        //*index = self.adj_store.len() - 1;
        } else {
            self.adj_store.push(AdjNode::first(to));
        }
        self.adj_index.insert(from, self.adj_store.len() - 1);
    }

    fn neighbors(&self, node: u32) -> Option<impl Iterator<Item = u32> + '_> {
        let node = match self.adj_index.get(&node) {
            Some(index) => Some(&self.adj_store[*index]),
            None => return None,
        };
        Some(AdjList::new(node, &self.adj_store))
    }

    #[inline]
    fn nodes(&self) -> impl Iterator<Item = &u32> {
        self.adj_index.keys()
    }

    pub fn find(&self) -> Result<(), String> {
        let mut search = Search::new(self.node_count(), 42);
        for node in self.nodes() {
            self.walk_graph(*node, &mut search)?;
            search.clear();
        }
        println!("Explored nodes: {}", search.node_explored);
        println!("Found cycles: {}", search.solutions.len());
        for sol in search.solutions {
            println!("Solution: {:x?}", sol.nonces);
        }
        Ok(())
    }

    fn add_solution(&self, s: &mut Search) -> Result<(), String> {
        let res: Result<Vec<_>, _> = s.path[s.path.len() - s.length..]
            .chunks(2)
            .map(|pair| match pair {
                &[n1, n2] => self.get_nonce(n1, n2),
                _ => Err("not an edge".to_string()),
            })
            .collect();
        let mut nonces = match res {
            Ok(v) => v,
            Err(e) => {
                return Err(format!("Failed to get nonce {:?}", e));
            }
        };
        nonces.sort();
        let sol = Solution { nonces };
        s.solutions.push(sol);
        Ok(())
    }

    fn walk_graph(&self, current: u32, search: &mut Search) -> Result<(), String> {
        if search.is_explored(current) {
            if search.is_cycle(current) {
                self.add_solution(search)?;
                println!("Found {}", current);
            }
            return Ok(());
        }

        let neighbors = match self.neighbors(current) {
            None => return Ok(()),
            Some(it) => it,
        };
        search.explore(current);
        for ns in neighbors {
            if !search.is_visited(ns) {
                search.visit(ns);
                //search.visit_edge();
                self.walk_graph(ns ^ 1, search)?;
                search.leave();
            } else {
                if search.is_cycle(ns) {
                    self.add_solution(search)?;
                }
            }
        }
        search.leave();
        Ok(())
    }
}
