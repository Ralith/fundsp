use super::math::*;
use super::*;
use generic_array::sequence::*;
use numeric_array::typenum::*;
use std::marker::PhantomData;

/// Type-level integer.
pub trait Size<T>: numeric_array::ArrayLength<T> {}

impl<T, A: numeric_array::ArrayLength<T>> Size<T> for A {}

/// Frames transport audio data between AudioNodes.
pub type Frame<T, Size> = numeric_array::NumericArray<T, Size>;

/// AudioNode processes audio data sample by sample.
/// It has a static number of inputs and outputs known at compile time.
pub trait AudioNode: Clone {
    type Inputs: Unsigned;
    type Outputs: Unsigned;

    /// Resets the input state of the component to an initial state where it has not processed any samples.
    /// In other words, resets time to zero.
    fn reset(&mut self, _sample_rate: Option<f64>) {}

    /// Processes one sample.
    fn tick<T: Float>(&mut self, input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>;

    /// Causal latency from input to output, in (fractional) samples.
    /// After a reset, we can discard this many samples from the output to avoid incurring a pre-delay.
    /// This applies only to components that have both inputs and outputs; others should return None.
    /// The latency can depend on the sample rate and is allowed to change after a reset.
    fn latency(&self) -> Option<f64> {
        // Default latency is zero.
        if self.inputs() > 0 && self.outputs() > 0 {
            Some(0.0)
        } else {
            None
        }
    }

    /// Ping contained nodes to obtain a deterministic pseudorandom seed.
    /// The local hash includes children, too.
    fn ping(&mut self, hash: u32) -> u32;

    // End of interface. There is no need to override the following.

    /// Number of inputs.
    #[inline]
    fn inputs(&self) -> usize {
        Self::Inputs::USIZE
    }

    /// Number of outputs.
    #[inline]
    fn outputs(&self) -> usize {
        Self::Outputs::USIZE
    }

    /// Retrieves the next mono sample from an all-zero input.
    /// If there are many outputs, chooses the first.
    /// This is an infallible convenience method.
    #[inline]
    fn get_mono<T: Float>(&mut self) -> T
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        assert!(self.outputs() >= 1);
        let output = self.tick(&Frame::default());
        output[0]
    }

    /// Retrieves the next stereo sample pair (left, right) from an all-zero input.
    /// If there are more outputs, chooses the first two. If there is just one output, duplicates it.
    /// This is an infallible convenience method.
    #[inline]
    fn get_stereo<T: Float>(&mut self) -> (T, T)
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        assert!(self.outputs() >= 1);
        let output = self.tick(&Frame::default());
        (output[0], output[if self.outputs() > 1 { 1 } else { 0 }])
    }

    /// Filters the next mono sample.
    /// Broadcasts the input to as many channels as are needed.
    /// If there are many outputs, chooses the first.
    /// This is an infallible convenience method.
    #[inline]
    fn filter_mono<T: Float>(&mut self, x: T) -> T
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        assert!(self.outputs() >= 1);
        let output = self.tick(&Frame::splat(x));
        output[0]
    }

    /// Filters the next stereo sample pair.
    /// Broadcasts the input by wrapping to as many channels as are needed.
    /// If there are more outputs, chooses the first two. If there is just one output, duplicates it.
    /// This is an infallible convenience method.
    #[inline]
    fn filter_stereo<T: Float>(&mut self, x: T, y: T) -> (T, T)
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        assert!(self.outputs() >= 1);
        let output = self.tick(&Frame::generate(|i| if i & 1 == 0 { x } else { y }));
        (output[0], output[if self.outputs() > 1 { 1 } else { 0 }])
    }
}

/// Combined latency of parallel components a and b.
fn parallel_latency(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(min(x, y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        _ => None,
    }
}

/// Combined latency of serial components a and b.
fn serial_latency(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        _ => None,
    }
}

/// PassNode passes through its inputs unchanged.
#[derive(Clone)]
pub struct PassNode<N> {
    _marker: PhantomData<N>,
}

impl<N> PassNode<N> {
    pub fn new() -> Self {
        PassNode {
            _marker: PhantomData,
        }
    }
}

impl<N: Unsigned> AudioNode for PassNode<N> {
    type Inputs = N;
    type Outputs = N;

    #[inline]
    fn tick<T: Float>(&mut self, input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        input.clone()
    }

    #[inline]
    // TODO: Find a clever way to do this automatically.
    fn ping(&mut self, hash: u32) -> u32 {
        hashw(0x001 ^ hash)
    }
}

/// SinkNode consumes its inputs.
#[derive(Clone)]
pub struct SinkNode<N> {
    _marker: PhantomData<N>,
}

impl<N> SinkNode<N> {
    pub fn new() -> Self {
        SinkNode {
            _marker: PhantomData,
        }
    }
}

impl<N: Unsigned> AudioNode for SinkNode<N> {
    type Inputs = N;
    type Outputs = U0;

    #[inline]
    fn tick<T: Float>(&mut self, _input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        Frame::default()
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        hashw(0x002 ^ hash)
    }
}

/// ConstantNode outputs a constant value.
#[derive(Clone)]
pub struct ConstantNode<T: Float, N: Size<T>> {
    output: Frame<T, N>,
}

impl<T: Float, N: Size<T>> ConstantNode<T, N> {
    pub fn new(output: Frame<T, N>) -> Self {
        ConstantNode { output }
    }
}

impl<T: Float, N: Size<T>> AudioNode for ConstantNode<T, N> {
    type Inputs = U0;
    type Outputs = N;

    #[inline]
    fn tick<U: Float>(&mut self, _input: &Frame<U, Self::Inputs>) -> Frame<U, Self::Outputs>
    where
        Self::Inputs: Size<U>,
        Self::Outputs: Size<U>,
    {
        Frame::generate(|i| convert(self.output[i]))
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        hashw(0x003 ^ hash)
    }
}

#[derive(Clone)]
pub enum Binop {
    Add,
    Sub,
    Mul,
}

pub trait FrameBinop: Clone {
    fn binop<T: Float, N: Size<T>>(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N>;
}
#[derive(Clone)]
pub struct FrameAdd;

impl FrameAdd {
    pub fn new() -> FrameAdd {
        FrameAdd
    }
}

impl FrameBinop for FrameAdd {
    #[inline]
    fn binop<T: Float, N: Size<T>>(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x + y
    }
}

#[derive(Clone)]
pub struct FrameSub;

impl FrameSub {
    pub fn new() -> FrameSub {
        FrameSub
    }
}

impl FrameBinop for FrameSub {
    #[inline]
    fn binop<T: Float, N: Size<T>>(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x - y
    }
}

#[derive(Clone)]
pub struct FrameMul;

impl FrameMul {
    pub fn new() -> FrameMul {
        FrameMul
    }
}

impl FrameBinop for FrameMul {
    #[inline]
    fn binop<T: Float, N: Size<T>>(x: &Frame<T, N>, y: &Frame<T, N>) -> Frame<T, N> {
        x * y
    }
}

#[derive(Clone)]
pub enum Unop {
    Neg,
}

pub trait FrameUnop: Clone {
    fn unop<T: Float, N: Size<T>>(x: &Frame<T, N>) -> Frame<T, N>;
}
#[derive(Clone)]
pub struct FrameNeg;

impl FrameNeg {
    pub fn new() -> FrameNeg {
        FrameNeg
    }
}

impl FrameUnop for FrameNeg {
    #[inline]
    fn unop<T: Float, N: Size<T>>(x: &Frame<T, N>) -> Frame<T, N> {
        -x
    }
}

/// BinopNode reduces a set of channels blockwise with a binary operation
#[derive(Clone)]
pub struct BinopNode<B, N> {
    _marker: PhantomData<N>,
    b: B,
}

impl<B, N> BinopNode<B, N> {
    pub fn new(b: B) -> Self {
        let node = BinopNode {
            _marker: PhantomData,
            b,
        };
        node
    }
}

impl<B, N> AudioNode for BinopNode<B, N>
where
    B: FrameBinop,
    N: Unsigned + Mul<U2> + Clone,
    Prod<N, U2>: Unsigned,
{
    type Inputs = Prod<N, U2>;
    type Outputs = N;

    #[inline]
    fn tick<T: Float>(&mut self, input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        let (x, y) = input.split_at(N::USIZE);
        B::binop(x.into(), y.into())
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        hashw(0x004 ^ hash)
    }
}

/// UnopNode applies an unary operator to its inputs.
#[derive(Clone)]
pub struct UnopNode<X, U> {
    x: X,
    u: U,
}

impl<X, U> UnopNode<X, U>
where
    X: AudioNode,
    U: FrameUnop,
{
    pub fn new(x: X, u: U) -> Self {
        let mut node = UnopNode { x, u };
        node.ping(0x0002);
        node
    }
}

impl<X, U> AudioNode for UnopNode<X, U>
where
    X: AudioNode,
    U: FrameUnop,
{
    type Inputs = X::Inputs;
    type Outputs = X::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
    }
    #[inline]
    fn tick<T: Float>(&mut self, input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        U::unop(&self.x.tick(input))
    }
    fn latency(&self) -> Option<f64> {
        self.x.latency()
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        let hash = self.x.ping(hash);
        hashw(0x005 ^ hash)
    }
}

#[derive(Clone)]
pub struct Map<T, F, I, O> {
    f: F,
    _marker: PhantomData<(T, I, O)>,
}

impl<T, F, I, O> Map<T, F, I, O>
where
    T: Float,
    F: Clone + FnMut(&Frame<T, I>) -> Frame<T, O>,
    I: Size<T>,
    O: Size<T>,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: PhantomData,
        }
    }
}

impl<T, F, I, O> AudioNode for Map<T, F, I, O>
where
    T: Float,
    F: Clone + FnMut(&Frame<T, I>) -> Frame<T, O>,
    I: Size<T>,
    O: Size<T>,
{
    type Inputs = I;
    type Outputs = O;

    // TODO: Implement reset() by storing initial state?

    #[inline]
    fn tick<U: Float>(&mut self, input: &Frame<U, Self::Inputs>) -> Frame<U, Self::Outputs>
    where
        Self::Inputs: Size<U>,
        Self::Outputs: Size<U>,
    {
        let out = (self.f)(&Frame::generate(|i| convert(input[i])));
        Frame::generate(|i| convert(out[i]))
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        hashw(0x007 ^ hash)
    }
}

/// PipeNode pipes the output of X to Y.
#[derive(Clone)]
pub struct PipeNode<X, Y> {
    x: X,
    y: Y,
}

impl<X, Y> PipeNode<X, Y>
where
    X: AudioNode,
    Y: AudioNode<Inputs = X::Outputs>,
{
    pub fn new(x: X, y: Y) -> Self {
        let mut node = PipeNode { x, y };
        node.ping(0x0003);
        node
    }
}

impl<X, Y> AudioNode for PipeNode<X, Y>
where
    X: AudioNode,
    Y: AudioNode<Inputs = X::Outputs>,
{
    type Inputs = X::Inputs;
    type Outputs = Y::Outputs;

    fn reset(&mut self, sample_rate: Option<f64>) {
        self.x.reset(sample_rate);
        self.y.reset(sample_rate);
    }
    #[inline]
    fn tick<T: Float>(&mut self, input: &Frame<T, Self::Inputs>) -> Frame<T, Self::Outputs>
    where
        Self::Inputs: Size<T>,
        Self::Outputs: Size<T>,
    {
        self.y.tick(&self.x.tick(input))
    }
    fn latency(&self) -> Option<f64> {
        serial_latency(self.x.latency(), self.y.latency())
    }
    #[inline]
    fn ping(&mut self, hash: u32) -> u32 {
        let hash = self.x.ping(hash);
        let hash = self.y.ping(hash);
        hashw(0x008 ^ hash)
    }
}

// //// StackNode stacks X and Y in parallel.
// #[derive(Clone)]
// pub struct StackNode<T, X, Y> {
//     _marker: PhantomData<T>,
//     x: X,
//     y: Y,
// }

// impl<T, X, Y> StackNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T>,
//     X::Inputs: Size<T> + Add<Y::Inputs>,
//     X::Outputs: Size<T> + Add<Y::Outputs>,
//     Y::Inputs: Size<T>,
//     Y::Outputs: Size<T>,
//     <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
//     <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
// {
//     pub fn new(x: X, y: Y) -> Self {
//         let mut node = StackNode {
//             _marker: PhantomData,
//             x,
//             y,
//         };
//         node.ping(0x0004);
//         node
//     }
// }

// impl<T, X, Y> AudioNode for StackNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T>,
//     X::Inputs: Size<T> + Add<Y::Inputs>,
//     X::Outputs: Size<T> + Add<Y::Outputs>,
//     Y::Inputs: Size<T>,
//     Y::Outputs: Size<T>,
//     <X::Inputs as Add<Y::Inputs>>::Output: Size<T>,
//     <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
// {
//     type Sample = T;
//     type Inputs = Sum<X::Inputs, Y::Inputs>;
//     type Outputs = Sum<X::Outputs, Y::Outputs>;

//     fn reset(&mut self, sample_rate: Option<f64>) {
//         self.x.reset(sample_rate);
//         self.y.reset(sample_rate);
//     }
//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let input_x = &input[0..X::Inputs::USIZE];
//         let input_y = &input[Self::Inputs::USIZE - Y::Inputs::USIZE..Self::Inputs::USIZE];
//         let output_x = self.x.tick(input_x.into());
//         let output_y = self.y.tick(input_y.into());
//         Frame::generate(|i| {
//             if i < X::Outputs::USIZE {
//                 output_x[i]
//             } else {
//                 output_y[i - X::Outputs::USIZE]
//             }
//         })
//     }
//     fn latency(&self) -> Option<f64> {
//         parallel_latency(self.x.latency(), self.y.latency())
//     }
//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         let hash = self.x.ping(hash);
//         let hash = self.y.ping(hash);
//         hashw(0x009 ^ hash)
//     }
// }

// /// BranchNode sends the same input to X and Y and concatenates the outputs.
// #[derive(Clone)]
// pub struct BranchNode<T, X, Y> {
//     _marker: PhantomData<T>,
//     x: X,
//     y: Y,
// }

// impl<T, X, Y> BranchNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T, Inputs = X::Inputs>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T> + Add<Y::Outputs>,
//     Y::Outputs: Size<T>,
//     <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
// {
//     pub fn new(x: X, y: Y) -> Self {
//         let mut node = BranchNode {
//             _marker: PhantomData,
//             x,
//             y,
//         };
//         node.ping(0x0005);
//         node
//     }
// }

// impl<T, X, Y> AudioNode for BranchNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T, Inputs = X::Inputs>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T> + Add<Y::Outputs>,
//     Y::Outputs: Size<T>,
//     <X::Outputs as Add<Y::Outputs>>::Output: Size<T>,
// {
//     type Sample = T;
//     type Inputs = X::Inputs;
//     type Outputs = Sum<X::Outputs, Y::Outputs>;

//     fn reset(&mut self, sample_rate: Option<f64>) {
//         self.x.reset(sample_rate);
//         self.y.reset(sample_rate);
//     }
//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let output_x = self.x.tick(input);
//         let output_y = self.y.tick(input);
//         Frame::generate(|i| {
//             if i < X::Outputs::USIZE {
//                 output_x[i]
//             } else {
//                 output_y[i - X::Outputs::USIZE]
//             }
//         })
//     }
//     fn latency(&self) -> Option<f64> {
//         parallel_latency(self.x.latency(), self.y.latency())
//     }
//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         let hash = self.x.ping(hash);
//         let hash = self.y.ping(hash);
//         hashw(0x00A ^ hash)
//     }
// }

// /// TickNode is a single sample delay.
// #[derive(Clone)]
// pub struct TickNode<T: Float, N: Size<T>> {
//     buffer: Frame<T, N>,
//     sample_rate: f64,
// }

// impl<T: Float, N: Size<T>> TickNode<T, N> {
//     pub fn new(sample_rate: f64) -> Self {
//         TickNode {
//             buffer: Frame::default(),
//             sample_rate,
//         }
//     }
// }

// impl<T: Float, N: Size<T>> AudioNode for TickNode<T, N> {
//     type Sample = T;
//     type Inputs = N;
//     type Outputs = N;

//     #[inline]
//     fn reset(&mut self, sample_rate: Option<f64>) {
//         if let Some(sample_rate) = sample_rate {
//             self.sample_rate = sample_rate;
//         }
//         self.buffer = Frame::default();
//     }

//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let output = self.buffer.clone();
//         self.buffer = input.clone();
//         output
//     }
//     fn latency(&self) -> Option<f64> {
//         Some(1.0 / self.sample_rate)
//     }
//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         hashw(0x00B ^ hash)
//     }
// }

// /// BusNode mixes together a set of nodes sourcing from the same inputs.
// #[derive(Clone)]
// pub struct BusNode<T, X, Y> {
//     _marker: PhantomData<T>,
//     x: X,
//     y: Y,
// }

// impl<T, X, Y> BusNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T, Inputs = X::Inputs, Outputs = X::Outputs>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T>,
//     Y::Inputs: Size<T>,
//     Y::Outputs: Size<T>,
// {
//     pub fn new(x: X, y: Y) -> Self {
//         let mut node = BusNode {
//             _marker: PhantomData,
//             x,
//             y,
//         };
//         node.ping(0x0006);
//         node
//     }
// }

// impl<T, X, Y> AudioNode for BusNode<T, X, Y>
// where
//     T: Float,
//     X: AudioNode<Sample = T>,
//     Y: AudioNode<Sample = T, Inputs = X::Inputs, Outputs = X::Outputs>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T>,
//     Y::Inputs: Size<T>,
//     Y::Outputs: Size<T>,
// {
//     type Sample = T;
//     type Inputs = X::Inputs;
//     type Outputs = X::Outputs;

//     fn reset(&mut self, sample_rate: Option<f64>) {
//         self.x.reset(sample_rate);
//         self.y.reset(sample_rate);
//     }
//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let output_x = self.x.tick(input);
//         let output_y = self.y.tick(input);
//         output_x + output_y
//     }
//     fn latency(&self) -> Option<f64> {
//         parallel_latency(self.x.latency(), self.y.latency())
//     }
//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         let hash = self.x.ping(hash);
//         let hash = self.y.ping(hash);
//         hashw(0x00C ^ hash)
//     }
// }

// /// FeedbackNode encloses a feedback circuit.
// /// The feedback circuit must have an equal number of inputs and outputs.
// #[derive(Clone)]
// pub struct FeedbackNode<T, X, N>
// where
//     T: Float,
//     X: AudioNode<Sample = T, Inputs = N, Outputs = N>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T>,
//     N: Size<T>,
// {
//     x: X,
//     // Current feedback value.
//     value: Frame<T, N>,
// }

// impl<T, X, N> FeedbackNode<T, X, N>
// where
//     T: Float,
//     X: AudioNode<Sample = T, Inputs = N, Outputs = N>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T>,
//     N: Size<T>,
// {
//     pub fn new(x: X) -> Self {
//         let mut node = FeedbackNode {
//             x,
//             value: Frame::default(),
//         };
//         node.ping(0x0007);
//         node
//     }
// }

// impl<T, X, N> AudioNode for FeedbackNode<T, X, N>
// where
//     T: Float,
//     X: AudioNode<Sample = T, Inputs = N, Outputs = N>,
//     X::Inputs: Size<T>,
//     X::Outputs: Size<T>,
//     N: Size<T>,
// {
//     type Sample = T;
//     type Inputs = N;
//     type Outputs = N;

//     #[inline]
//     fn reset(&mut self, sample_rate: Option<f64>) {
//         self.x.reset(sample_rate);
//         self.value = Frame::default();
//     }

//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let output = self.x.tick(&(input + self.value.clone()));
//         self.value = output.clone();
//         output
//     }

//     fn latency(&self) -> Option<f64> {
//         self.x.latency()
//     }

//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         let hash = self.x.ping(hash);
//         hashw(0x00D ^ hash)
//     }
// }

// /// FitNode adapts a filter to a pipeline.
// #[derive(Clone)]
// pub struct FitNode<X> {
//     x: X,
// }

// impl<X: AudioNode> FitNode<X> {
//     pub fn new(x: X) -> Self {
//         let mut node = FitNode { x };
//         node.ping(0x0008);
//         node
//     }
// }

// impl<X: AudioNode> AudioNode for FitNode<X> {
//     type Sample = X::Sample;
//     type Inputs = X::Inputs;
//     type Outputs = X::Inputs;

//     #[inline]
//     fn reset(&mut self, sample_rate: Option<f64>) {
//         self.x.reset(sample_rate);
//     }

//     #[inline]
//     fn tick(
//         &mut self,
//         input: &Frame<Self::Sample, Self::Inputs>,
//     ) -> Frame<Self::Sample, Self::Outputs> {
//         let output = self.x.tick(input);
//         Frame::generate(|i| {
//             if i < X::Outputs::USIZE {
//                 output[i]
//             } else {
//                 input[i]
//             }
//         })
//     }

//     fn latency(&self) -> Option<f64> {
//         self.x.latency()
//     }

//     #[inline]
//     fn ping(&mut self, hash: u32) -> u32 {
//         let hash = self.x.ping(hash);
//         hashw(0x00D ^ hash)
//     }
// }
