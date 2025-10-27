use std::marker::PhantomData;
use std::ptr::NonNull;

pub struct IntoIter<T, A> {

    // We use pub(super) here so that it is only public to this module
    // buf is a NonNull<T> here as we want to give a NonNull pointer to the start of the block
    // This does not move and, we use this to deallocate and, so we can reconstruct RawVec later to drop
    pub(super) buf: NonNull<T>,
    // Ptr, therefore, is mutable and is what we increment to move through the block.
    pub(super) ptr: NonNull<T>,
    // Specify the capacity
    pub(super) cap: usize,
    // We don't have a custom allocator but, we do need phantom data for the drop checker
    // So I combine the two - and later if we have an alloc we can add a separate field and get to keep the bounds
    pub(super) _alloc: PhantomData<A>,
    // We specify an end pointer which is the last element in the block offset by len
    pub(super) end: *const T,


}