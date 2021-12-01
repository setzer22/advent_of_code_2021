use crate::utils::*;
use itertools::Itertools;

#[test]
fn sonar_sweep() {
    let input_depths = read_int_lines("./inputs/01.txt").expect("Could open file");

    let mut prev = i32::MAX;
    let mut num_increases = 0;

    for depth in input_depths {
        if depth > prev {
            num_increases += 1;
        }
        prev = depth;
    }

    println!("Height increased a total of {} times", num_increases);
    assert_eq!(num_increases, 1624);
}

#[test]
fn sonar_sweep_window() {
    let input_depths = read_int_lines("./inputs/01.txt").expect("Could open file");

    let mut prev_sum = i32::MAX;
    let mut num_increases = 0;

    for (h1, h2, h3) in input_depths.tuple_windows() {
        let curr_sum = h1 + h2 + h3;
        if curr_sum > prev_sum {
            num_increases += 1;
        }
        prev_sum = curr_sum;
    }

    println!("Height increased a total of {} times", num_increases);
    assert_eq!(num_increases, 1653);
}
