#![feature(allocator_api)]
#![feature(slice_ptr_get)]        // often used in RawVec internals
#![feature(alloc_layout_extra)]   // sometimes needed for Layout::array

pub mod myvec;