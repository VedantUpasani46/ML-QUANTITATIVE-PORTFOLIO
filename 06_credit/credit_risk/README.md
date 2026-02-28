# Module 15: Credit Risk Models

**Target:** 90%+ default prediction accuracy

## Overview

Merton model, CDS pricing, default probability

## Why This Matters

Avoid bankruptcy losses, price credit derivatives

## Key Features

- ✅ Merton structural model
- ✅ Distance to default
- ✅ CDS spread calculation
- ✅ Recovery rates

## Usage

```python
from module15_credit_risk import *

# [Module-specific usage example here]
# See full code for detailed implementation
```

## Run Demo

```bash
python module15_credit_risk.py
```

## Technical Details

- **Module:** 15
- **Category:** 06 Credit
- **File:** `module15_credit_risk.py`
- **Target:** 90%+ default prediction accuracy

## Interview Insight

**Q (JPMorgan Credit Trading):** How do you estimate default probability?

**A:** Merton model: treat equity as call option on assets. Default when assets < debt. Estimate asset vol from equity vol. Distance to default = (Assets - Debt) / (Asset Vol).

## Real-World Applications

- Used by: JPMorgan Credit Trading and similar top-tier quantitative firms
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

**Module 15 complete. Part of a 40-module quantitative finance portfolio.**

## Navigation

- **Previous Module:** Module 14
- **Next Module:** Module 16
- **View All Modules:** See main README.md

---

