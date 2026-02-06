// benches/hqs_bench.rs
// ============================================================================
// HQS Benchmark
// ============================================================================

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use helios_convert::hqs::{quantize_hq5k, quantize_hq5k_fast, quantize_hq4k, quantize_hq4k_fast};
use rand::Rng;

fn generate_random_tensor(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-2.0..2.0)).collect()
}

fn bench_hq5k(c: &mut Criterion) {
    let mut group = c.benchmark_group("HQ5K");
    
    for size in [1024, 10240, 102400, 1048576].iter() {
        let tensor = generate_random_tensor(*size);
        
        group.bench_with_input(
            BenchmarkId::new("MSE", size),
            &tensor,
            |b, t| b.iter(|| black_box(quantize_hq5k(t))),
        );
        
        group.bench_with_input(
            BenchmarkId::new("Fast", size),
            &tensor,
            |b, t| b.iter(|| black_box(quantize_hq5k_fast(t))),
        );
    }
    
    group.finish();
}

fn bench_hq4k(c: &mut Criterion) {
    let mut group = c.benchmark_group("HQ4K");
    
    for size in [1024, 10240, 102400, 1048576].iter() {
        let tensor = generate_random_tensor(*size);
        
        group.bench_with_input(
            BenchmarkId::new("MSE", size),
            &tensor,
            |b, t| b.iter(|| black_box(quantize_hq4k(t))),
        );
        
        group.bench_with_input(
            BenchmarkId::new("Fast", size),
            &tensor,
            |b, t| b.iter(|| black_box(quantize_hq4k_fast(t))),
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_hq5k, bench_hq4k);
criterion_main!(benches);
