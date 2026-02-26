# @wlearn/lightgbm

LightGBM WASM port for wlearn. Gradient boosting for classification and regression,
running in browser and Node.js via WebAssembly.

## Installation

```bash
npm install @wlearn/lightgbm
```

## Usage

```javascript
import { LGBModel } from '@wlearn/lightgbm'

// Create and train
const model = await LGBModel.create({
  objective: 'binary',
  learning_rate: 0.05,
  num_leaves: 31,
  numRound: 100
})
model.fit(X, y)

// Predict
const predictions = model.predict(X_test)
const probabilities = model.predictProba(X_test)
const accuracy = model.score(X_test, y_test)

// Save and load
const bundle = model.save()
const loaded = await LGBModel.load(bundle)

// Clean up
model.dispose()
```

## Supported objectives

- `binary` -- binary classification
- `multiclass` -- multiclass classification (softmax)
- `multiclassova` -- multiclass one-vs-all
- `cross_entropy` -- cross-entropy classification
- `regression` -- regression (default)

## License

MIT (upstream LightGBM is MIT-licensed)
