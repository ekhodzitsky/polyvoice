fn main() {
    println!("cargo:rustc-check-cfg=cfg(linux_cblas)");
    if std::env::var("CARGO_CFG_TARGET_VENDOR").ok().as_deref() == Some("apple") {
        println!("cargo:rerun-if-changed=src/bnns_conv.c");
        println!("cargo:rerun-if-changed=src/bnns_graph.c");
        cc::Build::new()
            .file("src/bnns_conv.c")
            .file("src/bnns_graph.c")
            .flag("-fno-objc-arc")
            .compile("pv_bnns_conv");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(feature = "system-openblas")]
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux") {
        // Only the LP64 OpenBLAS ABI provides the i32 CBLAS arguments and
        // openblas_set_num_threads used by linux_cblas. Generic BLAS and
        // ILP64 openblas64 are not interchangeable with this interface.
        if let Err(error) = pkg_config::Config::new().probe("openblas") {
            eprintln!(
                "system-openblas requires LP64 OpenBLAS development files and pkg-config \
                 (for example, libopenblas-dev and pkg-config on Debian/Ubuntu). \
                 Set PKG_CONFIG_PATH for a non-system installation, or disable \
                 system-openblas to use Rust kernels.\n{error}"
            );
            std::process::exit(1);
        }
        println!("cargo:rustc-cfg=linux_cblas");
    }
}
