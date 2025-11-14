# TODO:
Fix generator issue:
Z1 and Z2 right now have the same generators, but that's not how it's supposed to be.
We need that Z2 has the generators of Z1 + the noise generators of partialZ.
On the upside this will make it unnecessary to explicitly construct Z2 based on the bounds...

Splitting and so on should stay the same

-> Integrate bound constraints into final LP check

## Simplest possible example
1 input dimension
x1 in [-1, 1]
eps in [-0.1, 0.1]
-> we are searching for an x2 in [-1,1] such that |x1-x2| in [-0.1,0.1]

x1 = 0.95*e1 + 0.05*e2

-> x1: Everything from -1 to 1 is reachable

eps = 0.1*e3

x2 = 0.95*e1 + 0.05*e4

### Overapproximation
x1  = 1.0 * e1
x2  = 1.0 * e2
eps = 0.1 * e3

very overapproximating:
x1 and x2 can take on completely independent values
eps just tells us that in practice values are at most eps apart

x1 = 1.0 * e1
x2 = 1.0 * e2 + 0.1 * e3
eps = 0.1 *e3

difference is exact, but we cover more area than we wish to for x2

x1 = 0.95 * e1 + 0.05 * e2
x2 = 0.95 * e1 + 0.05 * e3
eps = 0.05 * e2 - 0.05 * e3

If we split up e1:

x1 = 0.475 * e1 + 0.05 * e2 - 0.475
x2 = 0.475 * e1 + 0.05 * e3 - 0.475
eps = 0.05 * e2 - 0.05 * e3
--
x1 = 0.475 * e1 + 0.05 * e2 + 0.475
x2 = 0.475 * e1 + 0.05 * e3 + 0.475
eps = 0.05 * e2 - 0.05 * e3

If we split up e2 from above

x1 = 0.475 * e1 + 0.05 * e2 + 0.025
x2 = 0.475 * e1 + 0.05 * e3
eps = 0.025 * e2 - 0.05 * e3 + 0.025

x1 = 0.475 * e1 + 0.05 * e2 - 0.025
x2 = 0.475 * e1 + 0.05 * e3
eps = 0.025 * e2 - 0.05 * e3 - 0.025

Let x1, x2 in [-1,1] be such that |x1 - x2| <= 0.1
Case 1: |x1 - x2| <= 0.05
Case 1.1: x1 <= -0.95
  Set e1=-1, set e2=(x1+0.95)/0.05
  Then -0.95 + 0.05*(x1+0.95)/0.05 = x1
  ...

How would we split this? Let's say we want to split into x1<=0 / x1 >=0

Then: x1  in [-1,0]
      x2  in [-1,0.1]
      eps in [-0.1,0.1]
Find ``common center'' of x1 and x2
-> Find minimal, symmetric cut off for both such that same center
-> Need cut off of at least 0.1 for x2 then we would have [-0.9,0.0]

x2 = 0.55 * e_ - 0.45
   = 0.45 * e1 + 0.1  * e_ - 0.45
   = 0.45 * e1 + 0.05 * e3 + 0.05 * e4 - 0.45
x1 = 0.45 * e1 + 0.05 * e2 - 0.5 
xd = 

(a * e1 + b * e2 + c) - (d * e1 + e * e3 + f) <= 0.1
< - > (a - d) * e1 + b * e2 + e * e3 + (c - f) <= 0.1

Assume a,b,d,e>=0

- a - b + c = l1
  a + b + c = u1
- d - e + f = l2
  d + e + f = u2

  min((a-d),(d-a)) - b - e + (c - f) <= ld
  max((a-d),(d-a)) + b + e + (c - f) >= ud



# VeryDiff

VeryDiff is a tool for the equivalence verification of neural networks (NNs).
Given two NNs and a specification of an input region, it can answer the following question:

**Do the two NNs behave *equivalently* on the given input space?**

So far VeryDiff supports three different kinds of equivalence that can be checked for two NNs:
- $\varepsilon$-equivalence: The numerical outputs of the two NNs vary at most by $\varepsilon$ w.r.t. the $L_\infty$-norm
- Top-1 equivalence: The two NNs provide the same classification outputs
- $\delta$-Top-1 equivalence: If for some input the first NN provides classification $c$ with confidence larger $\delta > 0.5$, then the second NN also yields classification $c$.


The guarantees provided by VeryDiff are **sound**, i.e. if VeryDiff says two NNs are equivalent, they are provably so.
On the other hand, in some cases VeryDiff cannot be complete.
However, VeryDiff always tries to find counterexamples and outputs them if found.

## Installation
This software requires Julia 1.10.

Subsequently the software can be installed as follows:

```
git clone https://github.com/samysweb/VeryDiff-Release
cd VeryDiff-Release
./build.sh <path to julia binary>
```

On Linux `<path to julia binary>` can be found via `$(which julia)`.

## Running the tool

The binary of the tool can then be found in `./VeryDiff-Release/deps/VeryDiff/bin/VeryDiff`

Manual:
```
usage: <PROGRAM> [--epsilon EPSILON] [--top-1]
                 [--top-1-delta TOP-1-DELTA] [--timeout TIMEOUT]
                 [--naive] [-h] net1 net2 spec

positional arguments:
  net1                  First NN file (ONNX)
  net2                  Second NN file (ONNX)
  spec                  Input specification file (VNNLIB)

optional arguments:
  --epsilon EPSILON     Verify Epsilon Equivalence; provides the
                        epsilon value (type: Float64, default: -Inf)
  --top-1               Verify Top-1 Equivalence
  --top-1-delta TOP-1-DELTA
                        Verify δ-Top-1 Equivalence; provides the delta
                        value (type: Float64, default: -Inf)
  --timeout TIMEOUT     Timeout for verification (type: Int64,
                        default: 0)
  --naive               Use naive verification (without differential
                        verification)
  -h, --help            show this help message and exit
```