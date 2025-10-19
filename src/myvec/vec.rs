
// NOTE: Custom implementation of a vec using unsafe
#![allow(unstable_features)]

// A Vec can be comprised of 3 levels
// Level 1: Public facing Vec the object a user interacts with - holding a buf and len for how many items are allocated
// Level 2: RawVec which holds a RawInner, a Cap for storing the capacity of a vec and a marker for the drop checker
// Level 3: RawInnerVec which handles the generic memory allocation

use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ptr::{NonNull};

// For ZSTs there is zero allocation but a defined alignment
// A vec of Vec<()>::new() could be size = 0 but alignment = 1
// Allocators work in bytes so they give only non-zero-sized chunks of memory if we were to pass in a
// ZST the allocator would not know what to allocate but must provision for some memory so the trick is
// we let a capacity check handle this by setting capacity to 0..isize::MAX meaning set capacity to infinite
// as it doesn't matter because we are not actually allocating any actual memory

// For capacity handling, we need a type that stores capacity in a range that's safe for pointer arithmetic.
// In Rust's standard library, this is done using a niche optimization:
// the highest bit (the most significant bit) of a `usize` is always guaranteed to be zero
// because valid pointer offsets must be <= isize::MAX.
//
// Std can safely reserve that top bit as an unused niche.
// The compiler can then use that bit to represent `Option<Cap>::None` without
// increasing the size of the type (i.e., `Option<Cap>` stays the same size as `Cap`).
//
// In effect:
//   - All valid capacities are in 0..=isize::MAX (top bit = 0)
//   - `Option<Cap>::None` is represented by setting the top bit = 1
//   - If the top bit is ever set for a real capacity, we've overflowed and must fail
//
// This trick keeps `RawVec` compact and pointer-offset–safe without needing
// an extra tag field for `Option<Cap>`.

struct Cap(usize);

impl Cap {
	// SAFETY: cap must be <= isize::MAX
	const unsafe fn new_unchecked(cap: usize) -> Self {
		Self(cap)
	}
}

// Level 3
struct RawInnerVec<A: Allocator = Global> {
	//SAFETY: We have ownership of this raw byte buffer. It's untyped and safe for all T and can be
	// cast later. Is always NonNull
	ptr: NonNull<u8>, // The allocator allocated based on raw bytes and not on typed memory hence u8
	cap: usize, // Usize for now but will use custom Cap type to enforce 0..isize::MAX checks
	alloc: A,
}

// InnerVec will cast *mut u8 bytes to *mut T when using it

// Level 2
struct InnerVec<T, A: Allocator = Global> {
	inner: RawInnerVec<A>,
	_marker: PhantomData<T>, // To signal to the drop checker that there is a type here to check
}

// Level 1
struct MyVec<T, A: Allocator = Global> {
	buf: InnerVec<T, A>,
	len: usize,
}









#[cfg(test)]
mod tests {
	use super::*;
	#[test]
	fn test_simple() {
		println!("this is my vec");

		// Testing ZSTs
		let v: Vec<()> = Vec::new();
		println!("this is my vec {:?}", v.capacity());

	}
}