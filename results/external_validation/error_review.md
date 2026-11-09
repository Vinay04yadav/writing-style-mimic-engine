# External validation error review

This is a qualitative review of the four misclassified passages from the fixed external evaluation. It explains plausible sources of confusion; it does not establish their cause.

| Book | Chunk setting | True → predicted | Passage characteristics |
|---|---|---|---|
| *The Mysterious Affair at Styles* | 500 tokens | Christie → Austen | Courtroom proceedings and formal discussion of family relationships and inheritance. |
| *The Mysterious Affair at Styles* | 500 tokens | Christie → Doyle | Detective explanation about a will, evidence, and a mistaken inference. |
| *Great Expectations* | 500 tokens | Dickens → Doyle | Comic first-person dialogue with an unusual metaphor and legal allusion. |
| *Great Expectations* | 1,000 tokens | Dickens → Austen | Conversation about Miss Havisham, family background, wealth, and social class. |

These errors resemble the main internal confusion families: Christie/Doyle and Dickens/Austen. A few external passages also crossed between Dickens and Doyle. The passages appear to trigger on scene content, dialogue, and narrative form; this is a plausible explanation from manual inspection, not a causal finding about the classifier.

The cleaner's manifest records applied start boundaries for every external book, and these excerpts were inside the story bodies. No obvious Gutenberg front matter or author-label leakage appeared in the reviewed passages. Combining chunk scores still identified all ten external books correctly in both settings.

The external set remains excluded from training. Since these error examples have now been inspected, do not use this set repeatedly to tune future choices; add another untouched evaluation set if decisions are made from this review.
