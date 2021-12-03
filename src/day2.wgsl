struct Movement {
    dx: i32;
    dy: i32;
};

[[block]]
struct Input {
    data: array<Movement>;
};

[[block]]
struct Output {
    data: array<Movement>;
};

[[block]]
struct ItemsPerWorker {
    amt: u32;
};

[[binding(0), group(0)]]
var<storage, read> v_movements: Input;

[[binding(1), group(0)]]
var<storage, read_write> outputs: Output;

[[binding(2), group(0)]]
var<uniform> items_per_worker : ItemsPerWorker;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    //outputs.data[global_id.x] = i32(v_movements.data[global_id.x].dx) * i32(items_per_worker.amt);

    var idx = global_id.x;

    var i : u32 = 0u;
    var n : u32 = items_per_worker.amt;

    var dx_acc = 0;
    var dy_acc = 0;

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        dx_acc = dx_acc + v_movements.data[(idx * n) + i].dx; 
        dy_acc = dy_acc + v_movements.data[(idx * n) + i].dy; 
    }

    var output : Movement;
    output.dx = dx_acc;
    output.dy = dy_acc;
    outputs.data[idx] = output;
}