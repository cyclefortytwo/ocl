//extern crate ocl;
use ocl::builders::ProgramBuilder;
use ocl::{Device, Platform, ProQue, SpatialDims};
use std::env;

enum Mode {
    SetCnt = 1,
    Trim = 2,
    Extract = 3,
}
const RES_BUFFER_SIZE: usize = 2_000_000;
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

    let edge_bits = 29;
    let edge_count = 1024 * 1024 * 512 / 8; //1 << edge_bits;
    let node_count = 1024 * 1024 * 512 / 32; //edge_count * 2;
    let res_buf = vec![0; RES_BUFFER_SIZE];

    let mut prog_builder = ProgramBuilder::new();
    prog_builder.source_file("./src/lean.cl");

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
        result.map().enq()?;
    }
    pro_que.finish()?;

    println!("Done: size 0 {} 1 {} ", res_buf[0], res_buf[1]);
    Ok(())
}
