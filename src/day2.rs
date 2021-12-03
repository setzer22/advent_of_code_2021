// For day 2, let's do compute shaders! Note that this is just an exercise to
// learn how to interface with and run code on the GPU, for the input size and
// chosen implementation this algorithm would run more efficiently on the CPU.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::utils::*;

/// Each GPU worker will process this amount of items from the input.
/// Since there are 1000 elements, we will spawn 100 GPU workers.
const ITEMS_PER_WORKER: u32 = 10;

/// The movement struct encodes the lines from the input
/// #[repr(C)] is needed to make sure we get a consistent byte layout to send to GPU
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Movement {
    /// The amount of movement we get on the horizontal axis. Positive means forward.
    dx: i32,
    /// The amount of movement we get on the vertical axis. Positive means down.
    dy: i32,
}

// SAFETY: `Movement` is just two i32 values and marked #[repr(C)]
// These traits are a requirement to cast [Movement] to/from [u8]
unsafe impl bytemuck::Zeroable for Movement {}
unsafe impl bytemuck::Pod for Movement {}

impl Movement {
    /// Parses a `Movement` from one of the input lines
    pub fn from_line(line: impl AsRef<str>) -> Self {
        let line = line.as_ref();
        let mut split = line.split(" ");
        let dir = split.next().expect("First element is the direction");
        let amt = split
            .next()
            .expect("Second element is the amount")
            .parse::<i32>()
            .expect("Amount is integer");

        match dir {
            "forward" => Self { dx: amt, dy: 0 },
            "down" => Self { dx: 0, dy: amt },
            "up" => Self { dx: 0, dy: -amt },
            _ => panic!("Unexpected line: {}", line),
        }
    }
}

async fn run() {
    let lines = read_lines("./inputs/02.txt")
        .expect("Could parse")
        .map(|x| x.expect("Could parse line"))
        .map(Movement::from_line);

    let lines = lines.collect::<Vec<_>>();
    println!("{:?}", execute_gpu(lines.as_slice()).await);
}

async fn execute_gpu(input: &[Movement]) -> Option<i32> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    // `features` being the available features
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .ok()?;

    execute_gpu_inner(&device, &queue, input).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &[Movement],
) -> Option<i32> {
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("day2.wgsl"))),
    });

    // The input buffer will have one Movement entry per input line.
    let input_size = (input.len() * std::mem::size_of::<Movement>()) as wgpu::BufferAddress;

    // Used to store the problem input on the GPU and read it from the shader
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // The output buffer stores the problem result. Each worker will compute a
    // fraction of the output in parallel, and the final result will be
    // aggregated on the CPU.
    let output_size = (input.len() * std::mem::size_of::<Movement>() / ITEMS_PER_WORKER as usize)
        as wgpu::BufferAddress;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Used to read GPU results back into CPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // This uniform buffer is just to send a single u32 value, with the
    // ITEMS_PER_WORKER constant.
    let items_per_worker_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[ITEMS_PER_WORKER]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // There are two ways to define binding groups in wgpu. Implicitly or
    // explicitly. Implicit mode will introspect the shader code and generate an
    // appropiate bind group layout automatically.
    //
    // There are a few caveats when using this implicit API, though:
    //
    // - It's more difficult to know what the bind group layout is, from Rust
    //   alone. You need to look into the shaders.
    // - The introspection does not look at declarations but *usages*. If a
    //   storage buffer or uniform is unused in the shader code, the bind group
    //   layout returned will be incorrect. let bind_group_layout =
    //   compute_pipeline.get_bind_group_layout(0);
    //
    // Another option is to use the explicit API, i.e. what we do here. It
    // involves more boilerplate, but it is a more reliable method.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Input buffer (read only)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Items per worker uniform
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
    });

    // The pipeline layout is a way to specify multiple bind group layouts for a certain pipeline.
    // This is not necessary when using the implicit API
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // The compute pipeline. Describes which shader to run.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout), // This is set to None when using implicit API
        module: &cs_module,
        entry_point: "main",
    });


    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: items_per_worker_buffer.as_entire_binding(),
            },
        ],
    });

    // The encoder is what sends commands to the GPU
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("AoC 2021, day 2");
        // This signals the GPU to run the compute operation. You also specify
        // the "compute space" layout. We get up to 3 dimensions, but just need
        // one so we use 1 for the remaining two
        cpass.dispatch(input.len() as u32 / ITEMS_PER_WORKER, 1, 1);
    }

    // This tells the GPU to copy the buffer after the commpute shader is done.
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

    // This is where things are finally run
    queue.submit(Some(encoder.finish()));

    // What remains, is waiting for the commands to finish executing on the GPU
    // and fetch the data from our staging buffer
    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(_) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let results: Vec<Movement> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        println!("{:?}", &results[90..100]);

        let mut acc = (0, 0);
        for result in results {
            acc.0 += result.dx;
            acc.1 += result.dy;
        }

        Some(acc.0 * acc.1)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

#[test]
fn main() {
    env_logger::init();
    println!("The number is {:?}", pollster::block_on(run()));
}
