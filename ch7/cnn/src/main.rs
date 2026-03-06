fn main() {
    // Tiny dataset: 2 examples (6x6 grayscale), label 1 = cat, 0 = not cat
    let dataset = vec![
        (
            vec![
                vec![0.0, 1.0, 1.0, 0.0, 2.0, 3.0],
                vec![1.0, 2.0, 0.0, 1.0, 3.0, 1.0],
                vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
                vec![1.0, 0.0, 2.0, 3.0, 1.0, 0.0],
                vec![2.0, 3.0, 1.0, 0.0, 1.0, 2.0],
                vec![0.0, 1.0, 0.0, 2.0, 3.0, 1.0],
            ],
            1.0,
        ),
        (
            vec![
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ],
            0.0,
        ),
    ];
    let mut kernel = vec![
        vec![0.1, 0.2, -0.1],
        vec![0.0, 0.1, 0.1],
        vec![-0.2, 0.0, 0.2],
    ];

    let temp_conv = conv2d(&dataset[0].0, &kernel);
    let (temp_pool, _) = max_pool2x2(&temp_conv);
    let flat_len = temp_pool.len() * temp_pool[0].len();

    let mut fc_weights = vec![0.5; flat_len];
    let mut fc_bias = 0.0;
    let lr = 0.01;

    for epoch in 0..50 {
        let mut total_loss = 0.0;
        for (image, label) in &dataset {
            // --- Forward ---
            // Convolution + ReLU
            let mut conv_out = conv2d(image, &kernel);
            for i in 0..conv_out.len() {
                for j in 0..conv_out[0].len() {
                    conv_out[i][j] = relu(conv_out[i][j]);
                }
            }

            // Max pooling
            let (pool_out, max_pos) = max_pool2x2(&conv_out);

            // Flatten
            let flat = flatten(&pool_out);

            // Fully connected layer + sigmoid
            let z: f32 = flat
                .iter()
                .zip(fc_weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f32>()
                + fc_bias;
            let y_pred = sigmoid(z);

            // Compute loss
            let loss = binary_cross_entropy(*label, y_pred);
            total_loss += loss;

            // --- Backward ---
            let dz = y_pred - label;

            // Update Fc weights and bias
            for i in 0..fc_weights.len() {
                fc_weights[i] -= lr * dz * flat[i];
            }
            fc_bias -= lr * dz;

            // Backprop to pooled layer
            let mut d_pool_out = vec![vec![0.0; pool_out[0].len()]; pool_out.len()];
            let mut idx = 0;
            for i in 0..pool_out.len() {
                for j in 0..pool_out[0].len() {
                    d_pool_out[i][j] = dz * fc_weights[idx];
                    idx += 1;
                }
            }

            // Unpooling
            let d_conv_out =
                max_pool2x2_backprop(&d_pool_out, &max_pos, conv_out.len(), conv_out[0].len());

            // Backprop through ReLU
            let mut d_conv_out_relu = vec![vec![0.0; conv_out[0].len()]; conv_out.len()];
            for i in 0..conv_out.len() {
                for j in 0..conv_out[0].len() {
                    d_conv_out_relu[i][j] = d_conv_out[i][j] * relu_deriv(conv_out[i][j]);
                }
            }

            // Update convolution kernel
            conv2d_backprop(&d_conv_out_relu, image, &mut kernel, lr);
        }
        println!(
            "Epoch {}: Loss = {:.4}",
            epoch,
            total_loss / dataset.len() as f32
        );
    }

    println!("Trained kernel: {:?}", kernel);
    println!("Trained FC weights: {:?}", fc_weights);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn relu_deriv(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn binary_cross_entropy(y_true: f32, y_pred: f32) -> f32 {
    -(y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln())
}

fn conv2d(input: &[Vec<f32>], kernel: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let h = input.len();
    let w = input[0].len();
    let kh = kernel.len();
    let kw = kernel[0].len();

    let output_height = h - kh + 1;
    let output_width = w - kw + 1;

    let mut output = vec![vec![0.0; output_width]; output_height];

    for i in 0..output_height {
        for j in 0..output_width {
            let mut sum = 0.0;
            for k in 0..kh {
                for l in 0..kw {
                    sum += input[i + k][j + l] * kernel[k][l];
                }
            }
            output[i][j] = sum;
        }
    }

    output
}

fn conv2d_backprop(d_out: &[Vec<f32>], input: &[Vec<f32>], kernel: &mut [Vec<f32>], lr: f32) {
    let kh = kernel.len();
    let kw = kernel[0].len();
    for m in 0..kh {
        for n in 0..kw {
            let mut grad = 0.0;
            for i in 0..d_out.len() {
                for j in 0..d_out[0].len() {
                    grad += input[i + m][j + n] * d_out[i][j];
                }
            }
            kernel[m][n] -= lr * grad;
        }
    }
}

fn max_pool2x2(input: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<(usize, usize)>>) {
    let h = input.len() / 2;
    let w = input[0].len() / 2;
    let mut output = vec![vec![0.0; w]; h];
    let mut max_pos = vec![vec![(0, 0); w]; h];

    for i in 0..h {
        for j in 0..w {
            let mut max_val = f32::MIN;
            let mut max_idx = (0, 0);
            for m in 0..2 {
                for n in 0..2 {
                    let val = input[i * 2 + m][j * 2 + n];
                    if val > max_val {
                        max_val = val;
                        max_idx = (i * 2 + m, j * 2 + n);
                    }
                }
            }
            output[i][j] = max_val;
            max_pos[i][j] = max_idx;
        }
    }

    (output, max_pos)
}

fn max_pool2x2_backprop(
    d_out: &[Vec<f32>],
    max_pos: &[Vec<(usize, usize)>],
    h: usize,
    w: usize,
) -> Vec<Vec<f32>> {
    let mut d_input = vec![vec![0.0; w]; h];
    for i in 0..d_out.len() {
        for j in 0..d_out[0].len() {
            let (x, y) = max_pos[i][j];
            d_input[x][y] += d_out[i][j];
        }
    }
    d_input
}

fn flatten(input: &[Vec<f32>]) -> Vec<f32> {
    input.iter().flat_map(|r| r.clone()).collect()
}

fn predict(image: &[Vec<f32>], kernel: &[Vec<f32>], fc_weights: &[f32], fc_bias: f32) -> f32 {
    let mut conv_out = conv2d(image, kernel);
    for row in conv_out.iter_mut() {
        for val in row.iter_mut() {
            *val = relu(*val);
        }
    }
    let (pool_out, _) = max_pool2x2(&conv_out);
    let flat = flatten(&pool_out);
    let z: f32 = flat
        .iter()
        .zip(fc_weights.iter())
        .map(|(x, w)| x * w)
        .sum::<f32>()
        + fc_bias;

    sigmoid(z)
}
