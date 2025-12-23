import Base: push!, length, pop!
import Base.Order.lt
import Base.Order.Ordering
using DataStructures

struct VerificationTaskOrdering <: Ordering end
lt(o::VerificationTaskOrdering, a, b) = b.distance_bound < a.distance_bound # By distance to bound (probable violation first)
#b[1] < a[1] # By Workshare (largest first)
# b[2].distance_bound < a[2].distance_bound # By distance to bound (probable violation first)

mutable struct Queue
    queue::BinaryHeap{VerificationTask,VerificationTaskOrdering}
    function Queue()
        return new(BinaryHeap{VerificationTask}(VerificationTaskOrdering()))
    end
end

function empty!(q::Queue)
    q.queue = BinaryHeap{VerificationTask}(VerificationTaskOrdering())
end

function push!(q::Queue, x)
    push!(q.queue,x)
end
function pop!(q::Queue)
    return pop!(q.queue)
end
function length(q::Queue)
    return length(q.queue)
end

function peek_queue(q::Queue)
    return first(q.queue)
end