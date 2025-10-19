
// NOTE: Custom implementation of a vec using unsafe
#![allow(unstable_features)]

// A Vec can be comprised of 3 levels
// Level 1: Public facing Vec the object a user interacts with - holding a buf and len for how many items are allocated
// Level 2: RawVec which holds a RawInner, a Cap for storing the capacity of a vec and a marker for the drop checker
// Level 3: RawInnerVec which handles the generic memory allocation and layout specificity

use std::alloc::{Allocator, Global};
use std::marker::PhantomData;
use std::ptr::{NonNull};
use std::cmp::{Eq, PartialEq, Ord, PartialOrd};
use std::fmt::Alignment;
use crate::myvec::alignment::MyAlignment;
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

// We define a global capacity overflow panic to call whenever we encounter this
#[track_caller]
fn capacity_overflow() -> ! {
	panic!("capacity overflow");
}

// We use #![feature(rustc_attrs)] in lib.rs
// We do this so we can replicate the internally defined UsizeNoHighBit niche optimised type which reserves
// the top bit for the option tagging as mentioned above

// See [https://doc.rust-lang.org/src/alloc/raw_vec/mod.rs.html#40]

#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
#[rustc_layout_scalar_valid_range_start(0)]
#[rustc_layout_scalar_valid_range_end(0x7FFF_FFFF_FFFF_FFFF)]
struct CustomUsizeNoHighBit(usize);

impl CustomUsizeNoHighBit {

	pub const MAX: usize = isize::MAX as usize;

	// SAFETY: initializing type with `rustc_layout_scalar_valid_range` attr is unsafe

	pub const unsafe fn new(value: usize) -> Option<Self> {
		// SAFETY: We check the value here
		if value <= 0x7FFF_FFFF_FFFF_FFFF {
			Some(Self(value))
		} else {
			None
		}
	}

	/// Unsafe constructor.
	///
	/// # Safety
	/// Caller must guarantee `val <= isize::MAX`.
	pub const unsafe fn new_unchecked(val: usize) -> Self {
		Self(val)
	}

	pub const fn get(self) -> usize {
		self.0
	}

}

type Cap = CustomUsizeNoHighBit;

const ZERO_CAP: Cap = unsafe { CustomUsizeNoHighBit::new_unchecked(0) };

unsafe fn new_cap<T>(cap: usize) -> Cap {
	if std::mem::size_of::<T>() == 0 {
		ZERO_CAP
	} else {
		match unsafe { Cap::new(cap) } {
			Some(cap) => cap,
			None => capacity_overflow()
		}
	}
}


/// ## Level 3
///
/// # RawInnerVec
///
/// Is generic over the Allocator.
/// The standard library uses this separation for operations which require only the layout
///
/// We leave the type to the layer above in RawVec
struct RawInnerVec<A: Allocator = Global> {
	// We have ownership of this raw byte buffer. It's untyped and safe for all T and can be
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

// IMPL BLOCK for RawInnerVec

impl<A: Allocator> RawInnerVec<A> {
	#[inline]
	// A constructor usually does not allocate, so here we want to initialize and empty buffer with cap = 0
	// and a dangling pointer. We also need to store the alloc
	const fn new_in(alloc: A, align: MyAlignment) -> Self {
		// We need to define a NonNull pointer without provenance, meaning the pointer is cannot be used
		// for memory access as we have not yet allocated memory
		// NonNull::dangling() provides an abstracted function for this but the alignment is always known when we
		// have the T type.
		// For this layer, we pass down the alignment from the type so we must align ourselves
		let ptr = NonNull::without_provenance(_);
	}

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

	#[test]
	fn test_valid_capacity() {
		let c = unsafe { new_cap::<u8>(100) };
		assert_eq!(c.get(), 100);
	}

	#[test]
	fn test_zst_capacity() {
		struct Zst;
		let c = unsafe  { new_cap::<Zst>(100) };
		assert_eq!(c.get(), 0);
	}

	#[test]
	fn test_overflow_panics() {
		let too_big = (isize::MAX as usize) + 1;
		let result = std::panic::catch_unwind(|| unsafe { new_cap::<u8>(too_big) });
		assert!(result.is_err(), "expected panic for cap overflow");
	}

	#[test]
	fn test_max_value_ok() {
		let c = unsafe { new_cap::<u8>(isize::MAX as usize) };
		assert_eq!(c.get(), isize::MAX as usize);
	}

	#[test]
	fn test_option_niche_size() {
		use std::mem::size_of;
		assert_eq!(size_of::<Option<CustomUsizeNoHighBit>>(), size_of::<CustomUsizeNoHighBit>(),
		           "Option<CustomUsize> should be the same size (niche optimization active)");
	}

}