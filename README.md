# Project Mark Task 2 — Seasonal Lighting Cross-Version Matrix

This repository contains the exact pre-live candidate bytes and a GitHub Actions matrix used only to test the WeekOfYear admission condition across scikit-learn versions.

Admission condition:

`max(Sunday_%U, Monday_%W across tested versions) < 51.0 < min(ISO across tested versions)`

The workflow intentionally does not alter the candidate data, 51.0 gate, builder architecture, or WeekOfYear variants.
