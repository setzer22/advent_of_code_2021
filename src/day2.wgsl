struct Movement {
    dx: i32;
    dy: i32;
};

[[block]]
struct Day2Input {
    data: array<Movement>;
};

[[binding(0), group(0)]]
var<storage, read_write> v_movements: Day2Input;

//[[binding(1), group(0)]]
//var<storage, read_write> outputs: Day2Input;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    v_movements.data[global_id.x].dx = i32(global_id.x);
}