Here's the basic breakdown:

The input: 16 × 96
16 time frames × 96 features each. Each frame is 80ms of audio, so 16 frames = the last 1.28 seconds. The 96 numbers per frame come from the pre-trained openwakeword embedding model — they're a compressed fingerprint of what the audio sounds like at that moment.

Gemm — yes, matrix multiply. Called a "fully connected" or "dense" layer. Every input number is multiplied by a learned weight and summed into every output. It's how the network learns which combinations of features matter.

ReLU — Rectified Linear Unit. Much simpler than you guessed: f(x) = max(0, x). Negative? Zero it. Positive? Keep it. That's it. Without this, stacking matrix multiplies is mathematically pointless (they'd collapse into one). ReLU is what lets multiple layers actually learn different things.

Sigmoid — squashes any number into 0–1. That's your confidence score. 0 = definitely not the wake word, 1 = definitely is.

The whole thing in plain English:
Take 1.28 seconds of audio fingerprints → flatten into one long list of numbers → three rounds of "look for patterns, throw away negatives" → squeeze to a single probability.

The heavy lifting (turning raw audio into those 96-feature fingerprints) is done by the pre-trained embedding model upstream — your tiny 3-layer network just learns to recognise your wake word on top of it.

These are just bookkeeping ops — not learned, no weights.

Shape — reads the dimensions of the tensor (batch size, 16, 96). Used to make the next steps work dynamically regardless of batch size.

Concat — joins the shape info together into one list of numbers, e.g. [batch_size, 1536].

Reshape — uses that to flatten the 16×96 grid into a single vector of 1536 numbers per sample. That's what Gemm needs — one flat list, not a 2D grid.

So the three together are just: "flatten 16×96 into 1536 so we can feed it into the first fully connected layer."