#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,    // llama.h uses bool in extern "C" — fine on Linux/glibc
)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
