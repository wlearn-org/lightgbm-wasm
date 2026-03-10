# Changelog

## 0.2.0

- Wrap LGBModel with `createModelClass` for unified task detection
- Add `task` parameter: `'classification'` or `'regression'`, auto-detected from labels if omitted
- When both `task` and `objective` are set, `objective` takes precedence

## 0.1.0

- Initial release
- LightGBM v4.6.0 WASM port
- Binary classification, multiclass, regression
- Save/load via WLRN bundle format
- Extra trees mode support
