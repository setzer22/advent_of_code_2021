use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

pub fn read_lines(path: impl AsRef<Path>) -> io::Result<io::Lines<BufReader<File>>> {
    Ok(BufReader::new(File::open(path)?).lines())
}

pub fn read_int_lines(path: impl AsRef<Path>) -> io::Result<impl Iterator<Item = i32>> {
    Ok(read_lines(path)?.map(|x| {
        x.expect("Could parse line")
            .parse::<i32>()
            .expect("Line is i32")
    }))
}
