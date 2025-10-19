#![feature(rustc_attrs)]

#![feature(allocator_api)]
#![feature(slice_ptr_get)]        // often used in RawVec internals
#![feature(alloc_layout_extra)]
#![feature(ptr_alignment_type)]
// sometimes needed for Layout::array

extern crate core;

pub mod myvec;