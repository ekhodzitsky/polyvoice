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
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("linux") {
        // System OpenBLAS — same role as Accelerate on Apple. Missing pkg is
        // fine: GEMM stays on the in-crate kernel.
        let mut probed = pkg_config::Config::new();
        probed.cargo_metadata(true);
        if probed.probe("openblas").is_ok()
            || probed.probe("openblas64").is_ok()
            || probed.probe("blas").is_ok()
        {
            println!("cargo:rustc-cfg=linux_cblas");
        }
    }
}
