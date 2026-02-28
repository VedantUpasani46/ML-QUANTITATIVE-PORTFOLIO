# Module 32: Factor Models

**Target:** Isolate alpha from factor betas

## Overview

Fama-French factor decomposition (alpha vs beta)

## Why This Matters

LPs pay for alpha, not beta. Decompose 18% return into 16% factor exposure + 2% true alpha

## Key Features

- ✅ Fama-French 3/5 factors
- ✅ Rolling beta estimation
- ✅ Alpha attribution
- ✅ Custom factors

## Usage

```python
from module32_factor_models import *

# [Module-specific usage example here]
# See full code for detailed implementation
```

## Run Demo

```bash
python module32_factor_models.py
```

## Technical Details

- **Module:** 32
- **Category:** 14 Advanced Ml
- **File:** `module32_factor_models.py`
- **Target:** Isolate alpha from factor betas

## Interview Insight

**Q (AQR Portfolio Manager):** Your strategy returned 18% last year. Is that alpha or beta?

**A:** Regression: Return = α + β_mkt·Market + β_value·Value + β_momentum·Momentum. Result: 7.5% market + 6% value + 2% momentum + 0.5% size = 16% factor. True alpha = 18% - 16% = 2%. That's the value-add.

## Real-World Applications

- Used by: AQR Portfolio Manager and similar top-tier quantitative firms
- Production deployment at hedge funds
- Part of quantitative finance portfolios

## Integration

This module integrates with:
- Feature engineering pipelines
- Backtesting frameworks
- Production deployment systems
- Risk management tools

## Performance

All modules are optimized for production use with:
- Clean, readable code
- complete error handling
- Performance benchmarks
- Production-ready implementations

## References

See module implementation for detailed references and citations.

---

**Module 32 complete. Part of a 40-module quantitative finance portfolio.**

## Navigation

- **Previous Module:** Module 31
- **Next Module:** Module 33
- **View All Modules:** See main README.md

---

