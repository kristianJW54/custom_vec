
// NOTE: Custom implementation of a vec using unsafe
#![allow(unstable_features)]

// A Vec can be comprised of 3 levels
// Level 1: Public facing Vec the object a user interacts with - holding a buf and len for how many items are allocated
// Level 2: RawVec which holds a RawInner, a Cap for storing the capacity of a vec and a marker for the drop checker
// Level 3: RawInnerVec which handles the generic memory allocation and layout specificity

use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, GlobalAlloc, Layout};
use std::{alloc, cmp, fmt, mem, ptr, slice};
use std::marker::PhantomData;
use std::ptr::{NonNull};
use std::cmp::{Eq, PartialEq, Ord, PartialOrd};
use std::mem::ManuallyDrop;
use std::num::{NonZeroUsize};
use std::ops::{Deref, DerefMut};
use crate::myvec::into_iter::IntoIter;
// To try to follow with the std library implementation we will define a top level Alloc Enum which can
// help with telling the Allocator if we have un-initialized memory (basically do nothing) or if
// we have zeroed memory in which we need to allocate

#[derive(Clone, Copy, Debug)]
enum AllocInit {
	Uninitialized,
	Zeroed
}

// We define a global capacity overflow panic to call whenever we encounter this
#[track_caller]
fn capacity_overflow() -> ! {
	panic!("capacity overflow");
}

// We also need define some Layout helpers. A Layout represents the shape of a block of memory
// It's SIZE and ALIGNMENT
// How many bytes in total do we need to allocate
// and what is the alignment of each element.
// Essentially:
// We want to allocate memory for N elements of type T.
// Each element has a known size and alignment.
// Compute the total size (size * N) and keep the same alignment.
// If multiplication overflows, fail gracefully.
// Return that layout so we can tell the allocator how much to allocate.

// Example:
// Size = 4 - Alignment = 4
// 4 % 4 = 0 -> So we know we have good alignment
// 4 * 0 = 0 -> First memory address index is 0..4
// After 4 the next multiple is 8
// We add in the 4 bytes and this fills the [4..8] block perfectly
// After 8 the next multiple of 4 is 12
// Again this fits - so our blocks are [0..4][0..8][8..12] no padding needed
//
// Size = 6 - Alignment = 4
// 6 % 4 = 2 -> Not aligned meaning we need padding
// 4 * 0 = 0 -> First address is fine but the size 0..6 means space occupied will end at 6
// Next multiple after 6 is 8 [0..6] we need to pad 2 bytes [0..8] to align
// We add in the 6 bytes to the next block which becomes [8..14]
// The next multiple of 4 after 14 is 16 so we need to pad 2 bytes again [8..16]
// Our Layout so far is [0..6]+2 pad[8..14]+2 pad..etc.

// std achieves this with a helper using some nightly methods
// #[inline]
// fn layout_array(cap: usize, elem_layout: Layout) -> Result<Layout, TryReserveError> {
// 	elem_layout.repeat(cap).map(|(layout, _pad)| layout).map_err(|_| CapacityOverflow.into())
// }

fn layout_array_from_cap(cap: usize, elem_layout: Layout) -> Result<Layout, String> {

	// First need to check alignment and see if any padding is needed
	let pad = elem_layout.pad_to_align();
	// Then we need to check if it's ok to repeat this layout for size n and that we don't overflow and are
	// still aligned
	let size = pad
		.size()
		.checked_mul(cap)
		.ok_or_else(|| "capacity overflow")?;

	// Build the final layout
	Layout::from_size_align(size, pad.align()).map_err(|_| format!("invalid layout: size={} align={}", size, pad.align()))
}


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

// See [https://doc.rust-lang.org/src/alloc/raw_vec/mod.rs.html#40]
// Since we cannot replicate this in std without using internals, we must just take the
// Extra memory for Option<> instead of 8 bytes it will be 16 bytes.


#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
struct CustomUsize(usize);

impl CustomUsize {

	pub const MAX: usize = isize::MAX as usize;

	// SAFETY: initializing type with `rustc_layout_scalar_valid_range` attr is unsafe

	pub const unsafe fn new(value: usize) -> Option<Self> {
		// Safety: we checked upper bound.
		if value <= 0x7FFF_FFFF_FFFF_FFFF{
			Some(Self(value))
		} else {
			None
		}
	}

	/// Unsafe constructor.
	///
	///# Safety
	///Caller must guarantee `val <= isize::MAX`.
	pub const unsafe fn new_unchecked(value: usize) -> Self {
		Self(value)
	}

	pub const fn get(self) -> usize {
		self.0
	}

}

type Cap = CustomUsize;

const ZERO_CAP: Cap = unsafe { Cap::new_unchecked(0) };

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
///
/// At the moment, Allocator and Global is unstable so we can just use placeholder for now and continue
/// with global allocator
#[derive(Debug)]
struct RawInnerVec<A> {
	// We have ownership of this raw byte buffer. It's untyped and safe for all T and can be
	// cast later. Is always NonNull
	ptr: NonNull<u8>, // The allocator allocated based on raw bytes and not on typed memory hence u8
	cap: Cap, // Use custom Cap type to enforce 0..isize::MAX checks
	_alloc: PhantomData<A>,
}

// InnerVec will cast *mut u8 bytes to *mut T when using it

///# Level 2
///
/// InnerVec is the Typed Memory Manager - it adds T(type) awareness, knows layouts and ensures alignment
#[derive(Debug)]
struct InnerVec<T, A> {
	inner: RawInnerVec<A>,
	_marker: PhantomData<T>, // To signal to the drop checker that there is a type here to check
}

///# Level 1
///
/// MyVec is the top level vec which owns and tracks elements occupying the vec/block
struct MyVec<T, A> {
	buf: InnerVec<T, A>,
	len: usize,
}

// IMPL BLOCK for RawInnerVec
// #[global_allocator]
impl<A> RawInnerVec<A> {
	#[inline]
	// A constructor usually does not allocate, so here we want to initialize and empty buffer with cap = 0
	// and a dangling pointer. We also need to store the alloc
	const fn new_in(align: usize) -> Self {
		// We need to define a NonNull pointer without provenance, meaning the pointer cannot be used
		// for memory access as we have not yet allocated memory
		// NonNull::dangling() provides an abstracted function for this, which creates a NonNull<u8>

		// Std does this by using its internal Alignment implementation [https://doc.rust-lang.org/src/core/ptr/alignment.rs.html#13]
		// Looking through the docs it seems that for a RawInnerVec only an usize is needed which is derived from
		// using Alignment -> Alignment::of::<T>() which gives a NonZero<usize> to pass to Layout

		// align_of_val also gives this [https://doc.rust-lang.org/src/core/mem/mod.rs.html#509-512]
		// So to avoid using an internal structure and to avoid implementing my own Alignment struct
		// I will pass in an usize derived from align_of_val() which I can wrap in a NonZero

		//SAFETY: RawVec uses align_of_val which checks for correct alignment and non-zero so we are safe to
		// use new_unchecked here
		let nsu = unsafe { NonZeroUsize::new_unchecked(align) };
		let ptr = NonNull::dangling();
		// For alloc - we use PhantomData as placeholder for when Allocator internals become standard and we can
		// pass in a custom allocation for memory
		Self { ptr, cap: ZERO_CAP, _alloc: PhantomData }

	}

	fn with_capacity(cap: usize, elem_layout: Layout) -> Self {

		// TODO -> Need a try allocate in
		match Self::try_allocate(cap, AllocInit::Uninitialized, elem_layout) {
			Ok(this) => { this }
			Err(err) => { panic!("{}", err); }
		}

	}

	//TODO Need to make a try_allocate_in which looks to allocate a block of memory based on layout or alloc init and cap
	// If zst then we zero allocate
	// Need to check if the ptr returned from a zero_alloc points to memory or how is it handled?

	fn try_allocate(cap: usize, alloc_init: AllocInit, elem_layout: Layout) -> Result<Self, String> {
		// First we need to build and validate the layout
		let layout = match layout_array_from_cap(cap, elem_layout) {
			Ok(layout) => layout,
			Err(e) => return Err(e),
		};

		// If layout is 0 - we need to use new_in() method to make sure we don't allocate and instead return Self
		// which is a pointer without_prevenance() and points to no actual memory. This is done in the std library because
		// Drop is implemented and will not de-allocate on layouts/caps of 0 so if we allocate memory here we will not be able to release it
		if layout.size() == 0 {
			return Ok(Self::new_in(elem_layout.align()))
		};

		// Final check is for alloc_init - if we are uninitialized we are free to allocate memory
		// Otherwise if we are zeroed, then we use alloc_zeroed which allocates zero-initialized memory with the global allocator
		let result: NonNull<u8> = match alloc_init {
			AllocInit::Uninitialized => {
				// We use the alloc free functions specified for std stable - to_note: when the allocator trait becomes stable
				// we would use the A generic constraint on RawInnerVec to be generic over an allocator so we can use custom allocators
				// passed in that satisfy the trait

				//SAFETY: We have checked and validated our layout, that it is not zero and is aligned - we should be
				// safe to allocate from here - if not, we panic and handle alloc error
				let a = unsafe { alloc(layout) };
				if a.is_null() {
					handle_alloc_error(layout);
				};
				// SAFETY: We have already checked that a is NonNull
				unsafe { NonNull::new_unchecked(a) }
			}
			AllocInit::Zeroed => {

				//SAFETY: AllocInit is Zeroed so we know we can call alloc_zeroed here based on the enum
				let a = unsafe { alloc_zeroed(layout) };
				if a.is_null() {
					handle_alloc_error(layout);
				}
				// SAFETY: We have checked if NonNull
				unsafe { NonNull::new_unchecked(a) }
			},
		};

		Ok(Self { ptr: result, cap: unsafe { Cap::new_unchecked(cap) }, _alloc: PhantomData })

	}

	// Getters
	#[inline]
	const fn capacity(&self, elem_size: usize) -> usize {
		if elem_size == 0 { usize::MAX } else { self.cap.0 }
	}

	// Method to retrieve non_null ptr caste to the type
	const fn non_null_ptr<T>(&self) -> NonNull<T> {
		self.ptr.cast()
	}

	// Method to take a non_null ptr and return the *mut T ptr
	const fn ptr<T>(&self) -> *mut T {
		self.non_null_ptr().as_ptr()
	}

	#[inline]
	fn current_memory(&self, elem_layout: Layout) -> Option<(NonNull<u8>, Layout)> {

		if elem_layout.size() == 0 || self.cap.0 == 0 {
			None
		} else {
			// SAFETY: We have a valid layout with a capacity and pointer to allocated memory so we are safe to
			// build and fetch
			unsafe {
				// We get the allocation size by getting the size of the layout and multiplying it by the
				// capacity. Then we build the layout to return and fetch the pointer to the first
				// address in the block
				let alloc_size = elem_layout.size().unchecked_mul(self.cap.0);
				let layout = Layout::from_size_align_unchecked(alloc_size, elem_layout.align());
				Some((self.ptr.into(), layout))
			}
		}
	}

	// Memory methods

	// This deallocates the owned allocation - from std [https://doc.rust-lang.org/src/alloc/raw_vec/mod.rs.html#749]
	// This method is called by a Drop implementation

	// SAFETY: We own the allocation and can safely deallocate, we do not update teh ptr or capacity to prevent double-free and
	// use-after-free.
	unsafe fn deallocate(&mut self, elem_layout: Layout) {
		if let Some((ptr, layout)) = self.current_memory(elem_layout) {
			unsafe { dealloc(ptr.as_ptr(), layout) }
		}
	}

	// Grow Amortized
	// We separate grow_one and grow_amortized because a simple push call will only call grow_one with additional = 1 when
	// len == self.capacity()
	// Other methods such as extend or append may need to reserve more space and therefore will need to grow
	// to a larger block.
	fn grow_amortized(&mut self, len: usize, additional: usize, elem_layout: Layout) -> Result<(), String> {

		// Ensure that additional is larger than 0
		assert!(additional > 0);

		// Check that size is not 0 meaning we are not overfull
		if elem_layout.size() == 0 {
			return Err(String::from("capacity overflow"));
		}

		// We need to check if by adding the additional space we overflow or not
		let required_cap = len.checked_add(additional).ok_or(String::from("capacity overflow"))?;

		// We want to grow the cap so that it is either double or not less than what is immediately required
		// So if the current cap is 0 and, we require 1
		// Then if we tried to double - 0 * 2 = 0
		// We would have less than what is required - so we look at what is the min_non_zero cap as well
		// And take the max so that we have something that is not less than what is required and, we always grow exponentially

		// Take our current cap and double it - we then take the max of that or required cap
		let grow_cap = cmp::max(self.cap.0 * 2, required_cap);
		// We then take the max of either the min_non_zero or the grow_cap
		let grow_cap = cmp::max(min_non_zero_capacity(elem_layout.size()), grow_cap);

		// We need to build a new layout for the grown block
		let new_layout = layout_array_from_cap(grow_cap, elem_layout)?;

		// Now we need to allocate the memory - std does this using a finish_grow() function
		// [https://doc.rust-lang.org/src/alloc/raw_vec/mod.rs.html#766]
		// We will just do it here

		// We need to take the new layout and current memory and reallocate (if we can and have the contiguous space) or allocate new memory block
		// We then take the returned ptr and new cap and store them

		if let Some((ptr, old_layout)) = self.current_memory(elem_layout) {
			// std rightfully checks that old and new alignment are the same here to make sure
			// we are still allocating extra memory for the same aligned types
			assert_eq!(old_layout.align(), new_layout.align());
			unsafe {
				let new_ptr = alloc::realloc(ptr.as_ptr(), old_layout, new_layout.size());
				if new_ptr.is_null() {
					alloc::handle_alloc_error(new_layout);
				}

				self.ptr = NonNull::new_unchecked(new_ptr);
				self.cap = Cap::new_unchecked(grow_cap);
			}
			Ok(())

		} else {

			unsafe {
				let new_ptr = alloc::alloc(new_layout);
				if new_ptr.is_null() {
					alloc::handle_alloc_error(new_layout);
				};
				self.ptr = NonNull::new_unchecked(new_ptr);
				self.cap = Cap::new_unchecked(grow_cap);
			}
			Ok(())
		}

	}

	// Additional will always be 1 so we just need the layout
	// We also just panic for simplicity
	fn grow_one(&mut self, elem_layout: Layout) {
		match self.grow_amortized(elem_layout.size(), 1, elem_layout) {
			Ok(()) => return,
			Err(e) => { panic!("{}", e); }
		}
	}

}

//NOTE: std library uses two implement blocks for an InnerVec. The first being a Global allocator
// almost like a default, using the default global allocator and its functions
// The second being generic over A an Allocator which can be passed in.

// IMPL BLOCK for Default Global Allocator InnerVec
// For a default allocator we can use the free functions in alloc. [https://doc.rust-lang.org/alloc/alloc/index.html#functions]
// If we wanted to be generic over a future Allocator we could define a new impl block like std

// Before we get to the impl - std defines a help function for small vecs which are dumb - meaning
// as a minimum capacity of say 1 for certain types we should set a 'floor' for our capacity so as not to be wasteful
// for example, a u8 type with capacity of 1 will only take up 1 byte of memory but this ends up getting rounded up to about 8 bytes
// anyway and is cumbersome for meta-data. So it easier more efficient to just set a floor for minimum capacity

const fn min_non_zero_capacity(size: usize) -> usize {
	if size == 1 { // If we have an u8/i8 then we just round up to 8 bytes for minimum capacity
		8
	} else if size <= 1024 { // anything less than 1024 like u16 etc. should be 4 bytes
		4
	} else { // larger types should be exact
		1
	}
}

impl<T, A> InnerVec<T, A> {

	// public to this crate only
	pub(crate) const MIN_NON_ZERO_CAP: usize = min_non_zero_capacity(size_of::<T>());

	#[inline]
	pub fn new() -> Self {
		Self { inner: RawInnerVec::new_in(mem::align_of::<T>()), _marker: PhantomData }
	}

	#[inline]
	pub(crate) fn with_capacity(cap: usize) -> Self {
		Self { inner: RawInnerVec::with_capacity(cap, Layout::new::<T>()), _marker: PhantomData }
	}
	// Getters
	const fn capacity(&self) -> usize {
		self.inner.capacity(size_of::<T>())
	}

	const fn ptr(&self) -> *mut T {
		self.inner.ptr()
	}

	const fn non_null_ptr(&self) -> NonNull<T> {
		self.inner.non_null_ptr()
	}

	// Memory management

	// grow should be called when len == self.capacity() and will only be used when calling methods
	// such as push which are single element operations
	pub(crate) fn grow_one(&mut self) {
		self.inner.grow_one(Layout::new::<T>());
	}
}

//--------------------------------------
// Implementations for InnerVec

impl<T, A> Drop for InnerVec<T, A> {
	fn drop(&mut self) {
		println!("RawInnerVec deallocating buffer {:?}", self.ptr());
		// Because this is the second layer - above us is Vec which tracks the elements occupying the vec
		// Here we assume that vec has dropped those elements in place and leaves the deallocation of memory
		// to the below layers
		// As we are the memory manager we only need to call to inner to deallocate

		//SAFETY: We are the only ones who call to de-allocate here and do not leak
		// MyVec has already dropped elements at the addresses in the block so we are clear to remove the block
		unsafe { self.inner.deallocate(Layout::new::<T>()) }
	}
}


//--------------------------------------------
// Main Vec Impl Block
//--------------------------------------------

impl<T, A> MyVec<T, A> {

	pub fn new() -> Self {
		Self { buf: InnerVec::new(), len: 0 }
	}

	pub fn with_capacity(cap: usize) -> Self {
		Self { buf: InnerVec::with_capacity(cap), len: 0 }
	}

	pub fn push(&mut self, elem: T) {

		// First get the len
		let len = self.len;

		// We need to check if the vec needs to grow by comparing len == cap
		if len == self.buf.capacity() {
			// This grows the memory block and copies bytes over if new block had to be allocated
			self.buf.grow_one();
		}
		// SAFETY: If we reach here and don't panic on capacity overflow then we have a safe block of
		// memory with enough size to push another element (and more)
		unsafe {
			// Get a ptr to the start of the block
			let ptr = self.buf.ptr();
			// Add to the ptr the len so we can write to the end
			ptr.add(len).write(elem);
			// Then we update len to reflect new elem added
			self.len = len + 1;

			// std has a separate push_mut method which returns a ptr, but we omit this for now and
			// just use push() method
		}

	}

	pub fn as_ptr(&self) -> *const T {
		self.buf.ptr() as *const T
	}

	pub fn as_mut_ptr(&mut self) -> *mut T {
		self.buf.ptr()
	}

	pub fn non_null_ptr(&self) -> NonNull<T> {
		self.buf.non_null_ptr()
	}

	pub fn as_slice(&self) -> &[T] {
		// SAFETY: We own the ptr, track the len and manage the cap
		// We know that the memory block is valid and has sufficient space
		// We leverage the borrow checker by providing ownership and access through methods only
		// with &self || &mut self
		unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
	}

	pub fn as_mut_slice(&mut self) -> &mut [T] {
		unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
	}

}

//--------------------------------------------
// Main Vec Implementations
//--------------------------------------------

impl<T, A> Deref for MyVec<T, A> {
	type Target = [T];
	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<T, A> DerefMut for MyVec<T, A> {
	fn deref_mut(&mut self) -> &mut [T] {
		self.as_mut_slice()
	}
}

impl<T: fmt::Debug, A> fmt::Debug for MyVec<T, A> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Debug::fmt(&**self, f)
	}
}

impl<T, A> Drop for MyVec<T, A> {
	fn drop(&mut self) {
		println!("Dropping MyVec with len={} cap={}", self.len, self.buf.capacity());
		unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), self.len)) }
	}
	// RawVec handles de-allocation
}

// To use Iterator - we need to implement IntoIterator and for that we need to specify a custom IntoIter struct
// That we then implement Iterator for

// TODO -> Implement IntoIterator
impl<T, A> IntoIterator for MyVec<T, A> {
	type Item = T;
	type IntoIter = IntoIter<T, A>;
	fn into_iter(self) -> Self::IntoIter {

		let me = ManuallyDrop::new(self);

		let buf = me.buf.non_null_ptr();
		let ptr = me.buf.ptr();
		let end = if mem::size_of::<T>() == 0 {
			ptr.wrapping_byte_add(me.len)
		} else {
			unsafe { ptr.add(me.len) as *const T }
		};

		IntoIter { buf, ptr: buf, cap: me.buf.capacity(), _alloc: PhantomData, end }
	}
}

// The reason why this works is that we implemented Deref coercion on MyVec meaning
// any method available on &[T] becomes callable on &MyVec<T, A>
// Step by step
// 1. Self is of type &MyVec<T, A>
// 2. The compiler checks if Self has a .iter() method - it doesn't but &MyVec automatically deref's
//    Into &[T]
// 3. The compiler checks that &MyVec implements Deref - it does
// 4. Now Self has access to all of slices methods including .iter()

// Deref wouldn't be called with MyVec because the Trait is defined for &T signature

impl<'a, T, A> IntoIterator for &'a MyVec<T, A> {
	type Item = &'a T;
	type IntoIter = slice::Iter<'a, T>;

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}


#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_start() {

		let mut my_vec: MyVec<i32, ()> = MyVec::new();
		my_vec.push(10);

		println!("my_vec {:?}", my_vec);

		println!("my_vec sliced at index 0 -> [{:?}]", my_vec[0]);

	}

	#[test]
	fn test_layout_alignment_helper() {

		let cap = 10;
		let layout = layout_array_from_cap(cap, Layout::new::<i32>()).unwrap();

		let want = cap * 4;
		assert_eq!(layout.size(), want);
		assert_eq!(layout.align(), 4);

	}

	#[test]
	fn test_zst_capacity() {
		let mv: InnerVec<(), ()> = InnerVec::new();
		assert_eq!(mv.capacity(), 18446744073709551615);

	}

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
		assert_ne!(size_of::<Option<CustomUsize>>(), size_of::<CustomUsize>(),
				   "Option<CustomUsize> should be the same size (niche optimization active)");
	}

	#[test]
	fn test_inner_vec_new() {

		let v: InnerVec<i32, ()> = InnerVec::new();

		assert!(v.inner.cap.get() == 0);

	}

	#[test]
	fn test_raw_inner_vec_with_capacity_method() {
		let cap_wanted: usize = 10;

		// Normal (non-ZST) layout
		let layout = Layout::new::<i32>();
		let my_vec: RawInnerVec<()> = RawInnerVec::with_capacity(cap_wanted, layout);
		assert_eq!(my_vec.cap.0, 10);

		let addr = my_vec.ptr.as_ptr() as usize;
		assert_eq!(addr % layout.align(), 0, "pointer is not aligned");

		// ZST layout
		let zst_layout = Layout::new::<()>();
		let my_vec2: RawInnerVec<()> = RawInnerVec::with_capacity(cap_wanted, zst_layout);
		assert_eq!(my_vec2.cap.0, 0);

		// Expected bytes only for non-ZST
		let expected_bytes = layout.size() * cap_wanted;
		println!("Allocated {} bytes @ {:p}", expected_bytes, my_vec.ptr);

		// Optional "poke test" - only for non-zero-sized
		if expected_bytes > 0 {
			unsafe {
				std::ptr::write_bytes(my_vec.ptr.as_ptr(), 0xAB, expected_bytes);
			}
		}

		let zst_addr = my_vec2.ptr.as_ptr() as usize;
		assert!(zst_addr == 1 || zst_addr == usize::MAX, "ZST ptr should be dangling");
	}

	#[test]
	fn test_raw_inner_grow_amortized() {

		let layout = Layout::new::<()>();

		let mut vec: RawInnerVec<()> = RawInnerVec::with_capacity(4, layout);

		println!("{:?}", vec.current_memory(layout));

		vec.grow_amortized(2, 4, layout).expect_err("should panic");

		println!("{:?}", vec.current_memory(layout));

		let real_layout = Layout::new::<u8>();

		let mut real_vec: RawInnerVec<()> = RawInnerVec::with_capacity(4, real_layout);

		println!("Before: {:?}", real_vec.current_memory(real_layout));

		real_vec.grow_amortized(2, 4, real_layout).unwrap();

		println!("After: {:?}", real_vec.current_memory(real_layout));

	}

	#[test]
	fn test_my_vec_iter() {

		let mut my_vec: MyVec<i32, ()> = MyVec::new();
		my_vec.push(10);
		my_vec.push(20);
		my_vec.push(30);

		println!("{:?}", my_vec);

		for i in &my_vec {
			println!("{:?}", i);
		}

		std::mem::drop(my_vec);

	}

	#[test]
	fn test_change_element() {

		let mut my_vec: MyVec<i32, ()> = MyVec::new();
		my_vec.push(10);
		my_vec.push(20);
		my_vec.push(30);

		println!("{:?}", my_vec);

		my_vec[2] = 40;

		println!("{:?}", my_vec);

	}

}