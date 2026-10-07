//! The exact geometry judgements are made over.
//!
//! [`linalg`] is linear algebra over the field `Q(ζ_N)` on the vectors
//! states are; [`pauli`] is Pauli strings and the stabiliser codes they fix;
//! [`cover`] is covers and gauges, what makes a gauge one, and the topology
//! of the two: nerves, transition classes and orthogonality graphs.
//! [`sim`] computes what a circuit does to a state, exactly; [`stabilizer`]
//! does the same for a Clifford circuit over a code, in time polynomial in
//! the code's width, and decides restrictions and gauges on codes too wide
//! to write out; [`cert`] derives an operator's certificate from either,
//! and reads the contract classes, the frame rule and holonomies off
//! certificates; [`record`] is the judgement record an operator gets, and
//! what a unit publishes of it for its importers, whose claims are checked
//! again against it when a program is linked.
//!
//! Nothing here is approximate. Every question is decided by exact
//! arithmetic in the field, or refused.

pub mod cert;
pub mod cover;
pub mod linalg;
pub mod pauli;
pub mod record;
pub mod sim;
pub mod stabilizer;
