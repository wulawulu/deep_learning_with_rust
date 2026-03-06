fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use autodiff::*;
    use itertools::iterate;
    use plotters::prelude::*;
    use rand::RngExt;

    #[test]
    fn problem1() {
        let x: FT<f64> = FT::new(2.5, 1.0);
        let f = |x: FT<f64>| 3.0 * x.pow(2.0) + 2.0 * x + 1.0;

        let v: FT<f64> = f(x);

        println!("f = {}", v.x);
        println!("df = {}", v.dx);
    }

    #[test]
    fn problem2() {
        let f = |v: &[FT<f64>]| (v[0] - v[1]).pow(2.0);

        let df = grad(f, &vec![3.0, 2.0]);
        println!("Loss 对 y 的偏导: {}", df[0]);
        println!("Loss 对 y_hat 的偏导: {}", df[1]);
    }

    #[test]
    fn problem3() {
        let f = |v: &[FT<f64>]| relu(0.3 * v[0] + 0.6 * v[1] + 0.1);

        let df = grad(f, &vec![2.0, -1.0]);
        println!("Loss 对 x_1 的偏导: {}", df[0]);
    }

    fn relu(x: FT<f64>) -> FT<f64> {
        if x.x > 0.0 { x } else { FT::cst(0.0) }
    }

    #[test]
    fn problem4() -> Result<(), Box<dyn std::error::Error>> {
        // 从 -5.0 开始，每次加 0.1，取到 5.0 为止
        let points:Vec<(f64, f64)> = iterate(-5.0, |&x| x + 0.1)
            .take_while(|&x| x <= 5.0 + 1e-10) // 加个极小值防止浮点数精度问题
            .map(|x| (x, x.max(0.0))).collect(); // 计算 ReLU

        let root = BitMapBackend::new("training_loss.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;


        let mut chart = ChartBuilder::on(&root)
            .caption("Simulated Training Loss", ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(-5.0..5.0, 0.0..5.0)?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                points,
                &BLUE,
            ))?
            .label("Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        println!("Loss plot saved to training_loss.png");
        Ok(())
    }
}
