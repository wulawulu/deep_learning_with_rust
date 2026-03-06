use autodiff::*;

fn main() {
    let f = |v: &[FT<f64>]| v[0] * v[1].sin() + v[1] * v[1];

    let df = grad(f, &vec![1.0, 2.0]);

    println!("df/dx = {}", df[0]);
    println!("df/dy = {}", df[1]);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_grad1() {
        let f = |x: FT<f64>| x.sin() + x.exp();

        let df = diff(f, 1.0);
        println!("df = {}", df);
    }
}
