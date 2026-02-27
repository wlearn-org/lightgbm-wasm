# @wlearn/lightgbm

LightGBM v4.6.0 compiled to WebAssembly. Gradient boosting for classification and regression in browsers and Node.js.

Based on [LightGBM v4.6.0](https://github.com/microsoft/LightGBM) (MIT). Zero dependencies. ESM.

## Install

```bash
npm install @wlearn/lightgbm
```

## Quick start

```js
import { LGBModel } from '@wlearn/lightgbm'

const model = await LGBModel.create({
  objective: 'binary',
  learning_rate: 0.05,
  num_leaves: 31,
  numRound: 100
})

// Train -- accepts number[][] or { data: Float32Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])  // Float64Array

// Probabilities
const probs = model.predictProba([[2, 3], [6, 7]])  // Float64Array (nrow * nclass)

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
const buf = model.save()  // Uint8Array (WLRN bundle)
const model2 = await LGBModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## Typed matrix input

For best performance, pass pre-formatted typed arrays instead of `number[][]`:

```js
const X = {
  data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
  rows: 4,
  cols: 2
}
model.fit(X, [0, 0, 1, 1])
```

Note: LightGBM uses Float32 internally. If you pass Float64Array it will be converted.

## Task parameter

Instead of specifying LightGBM objective strings directly, you can use the unified `task` parameter:

```js
// These are equivalent:
await LGBModel.create({ task: 'classification' })  // auto-selects 'binary' or 'multiclass'
await LGBModel.create({ objective: 'binary' })

await LGBModel.create({ task: 'regression' })
await LGBModel.create({ objective: 'regression' })
```

When `task: 'classification'` is set, the objective is chosen automatically based on the number of unique labels in y: `binary` for 2 classes, `multiclass` for 3+.

Setting both `task` and `objective` throws an error.

## API

### `LGBModel.create(params?)`

Async factory. Loads WASM module, returns a ready-to-use model.

Parameters:
- `objective` -- LightGBM objective string (default: `'regression'`)
- `task` -- `'classification'` or `'regression'` (alternative to `objective`)
- `learning_rate` -- step size shrinkage (default: `0.1`)
- `num_leaves` -- max leaves per tree (default: `31`)
- `numRound` -- number of boosting rounds (default: `100`)
- `max_depth` -- max tree depth, -1 for no limit (default: `-1`)
- `subsample` -- row subsampling ratio (default: `1.0`)
- `colsample_bytree` -- column subsampling ratio (default: `1.0`)
- `min_child_weight` -- minimum sum of instance weight in a child (default: `1e-3`)
- `reg_lambda` -- L2 regularization (default: `0.0`)
- `reg_alpha` -- L1 regularization (default: `0.0`)
- `num_class` -- number of classes for multiclass (auto-set when using `task`)
- `verbosity` -- -1 = fatal, 0 = error, 1 = info (default: `-1`)

### `model.fit(X, y)`

Train on data. Returns `this`.
- `X` -- `number[][]` or `{ data: Float32Array, rows, cols }`
- `y` -- `number[]` or typed array

### `model.predict(X)`

Returns `Float64Array` of predicted labels (classification) or values (regression).

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major probabilities). Available for `binary`, `multiclass`, and `multiclassova` objectives.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()` / `LGBModel.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with LightGBM text model blob).

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `LGBModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Supported objectives

- `binary` -- binary classification
- `multiclass` -- multiclass classification (softmax)
- `multiclassova` -- multiclass one-vs-all
- `cross_entropy` -- cross-entropy classification
- `regression` -- regression (default)

All standard LightGBM objectives should work. These are tested in CI.

## Low-level API

For direct access to LightGBM's C API, use the lower-level `Dataset` and `Booster` classes:

```js
import { loadLGB, Dataset, Booster } from '@wlearn/lightgbm'

await loadLGB()

const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])
const ds = new Dataset(data, 4, 2, 'objective=binary verbosity=-1')
ds.setLabel(new Float32Array([0, 0, 1, 1]))

const booster = new Booster(ds.handle, 'objective=binary verbosity=-1')
for (let i = 0; i < 100; i++) {
  booster.update()
}

const preds = booster.predict(data, 4, 2)   // Float64Array
const modelBytes = booster.saveModel()        // Uint8Array (text format)

booster.dispose()
ds.dispose()
```

### `Dataset(data, nrow, ncol, params?)`

- `data` -- `Float32Array` (row-major)
- `params` -- LightGBM parameter string (`"key1=value1 key2=value2"`)
- `.setLabel(labels)` -- set target labels (`Float32Array`)
- `.dispose()` -- free WASM memory

### `Booster(trainDataHandle, paramsStr)`

- `.update()` -- run one training round, returns `true` if training finished
- `.predict(data, nrow, ncol, opts?)` -- predict, returns `Float64Array`
- `.saveModel()` -- returns `Uint8Array` (LightGBM text format)
- `.getNumClasses()` -- number of classes
- `.dispose()` -- free WASM memory

### `Booster.loadModel(buffer)`

Load from `Uint8Array`. Returns a `Booster`.

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every `Dataset`, `Booster`, and `LGBModel` when done. A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/lightgbm-wasm
cd lightgbm-wasm
bash scripts/build-wasm.sh
node --test test/
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## License

MIT (same as upstream LightGBM)
