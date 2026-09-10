# Shorthand module: meaning and scope

In this project, **shorthand means a reusable symbol for a relationship**,
defined in a legend and applied to different subjects and objects:

```text
Original: The cat is on the table. The book is on the shelf.

Legend: [x1] = the subject is on the object

cat [x1] table.
book [x1] shelf.
```

The model receives both the legend and the encoded text. Every stated fact
and qualification must retain its meaning. Summarization and exact repeated
phrase substitution are separate techniques, not interchangeable definitions
of shorthand.

This is the intended concept, not the literal output of the current codecs.
`relations_v1` is a narrow positive `is on` prototype using `@`.
`dictionary_v1` performs exact repeated-phrase substitution; `SummaryCodec`
is an optional externally supplied summarizer. The broader relational design
is not fully implemented or validated by the dictionary token-count test.

Read the [canonical definition, extended example, and implementation
boundaries](../../docs/prompt_codec.md#what-shorthand-means-in-this-project)
before extending this module. That document also covers installation, API
adapters, request-local scope, and evaluation including legend overhead.
