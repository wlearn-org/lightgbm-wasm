import { LGBModel, loadLGB, Dataset, Booster } from '../src/index.js'
import { decodeBundle } from '@wlearn/core'

// --- Test harness ---

let passed = 0
let failed = 0
let total = 0

async function test(name, fn) {
  total++
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}: ${err.message}`)
    failed++
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed')
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) {
    throw new Error(`${msg || ''} expected ${a} close to ${b} (tol=${tol}, diff=${Math.abs(a - b)})`)
  }
}

// --- Deterministic data generation (LCG PRNG) ---

function makeLCG(seed) {
  let state = seed
  return () => {
    state = (state * 1664525 + 1013904223) & 0x7fffffff
    return state / 0x7fffffff
  }
}

function makeBinaryData(n, seed = 42) {
  const rng = makeLCG(seed)
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2 - 1
    const x2 = rng() * 2 - 1
    X.push([x1, x2])
    y.push(x1 + x2 > 0 ? 1 : 0)
  }
  return { X, y }
}

function makeMulticlassData(n, nClasses = 3, seed = 42) {
  const rng = makeLCG(seed)
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2 - 1
    const x2 = rng() * 2 - 1
    const x3 = rng() * 2 - 1
    X.push([x1, x2, x3])
    // Simple class assignment based on dominant feature
    if (x1 > x2 && x1 > x3) y.push(0)
    else if (x2 > x3) y.push(1)
    else y.push(2)
  }
  return { X, y }
}

function makeRegressionData(n, seed = 42) {
  const rng = makeLCG(seed)
  const X = [], y = []
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2 - 1
    const x2 = rng() * 2 - 1
    X.push([x1, x2])
    y.push(3 * x1 + 2 * x2 + (rng() - 0.5) * 0.2)
  }
  return { X, y }
}

// --- Tests ---

console.log('\n=== @wlearn/lightgbm Test Suite ===\n')

console.log('-- WASM Loading --')

await test('WASM module loads', async () => {
  const wasm = await loadLGB()
  assert(wasm, 'WASM module should be truthy')
  assert(typeof wasm._wl_lgb_booster_create === 'function', 'booster_create should be a function')
})

await test('loadLGB is idempotent', async () => {
  const m1 = await loadLGB()
  const m2 = await loadLGB()
  assert(m1 === m2, 'should return same module instance')
})

console.log('\n-- Dataset --')

await test('Dataset creation from Float32Array', async () => {
  const data = new Float32Array([1, 2, 3, 4, 5, 6])
  const ds = new Dataset(data, 2, 3)
  assert(ds.handle, 'should have handle')
  ds.dispose()
})

await test('Dataset setLabel', async () => {
  const data = new Float32Array([1, 2, 3, 4])
  const ds = new Dataset(data, 2, 2)
  ds.setLabel(new Float32Array([0, 1]))
  ds.dispose()
})

await test('Dataset double dispose is safe', async () => {
  const data = new Float32Array([1, 2, 3, 4])
  const ds = new Dataset(data, 2, 2)
  ds.dispose()
  ds.dispose() // should not throw
})

console.log('\n-- Booster --')

await test('Booster create + update + predict', async () => {
  const { X, y } = makeBinaryData(100)
  const rows = X.length, cols = X[0].length
  const flat = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) flat[i * cols + j] = X[i][j]
  }

  const ds = new Dataset(flat, rows, cols, 'objective=binary verbosity=-1')
  ds.setLabel(new Float32Array(y))

  const booster = new Booster(ds.handle, 'objective=binary verbosity=-1 num_leaves=8')
  for (let i = 0; i < 10; i++) booster.update()

  const preds = booster.predict(flat, rows, cols)
  assert(preds.length === rows, `expected ${rows} predictions, got ${preds.length}`)
  assert(preds[0] >= 0 && preds[0] <= 1, 'binary predictions should be probabilities')

  ds.dispose()
  booster.dispose()
})

await test('Booster save and load', async () => {
  const { X, y } = makeBinaryData(50)
  const rows = X.length, cols = X[0].length
  const flat = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) flat[i * cols + j] = X[i][j]
  }

  const ds = new Dataset(flat, rows, cols, 'objective=binary verbosity=-1')
  ds.setLabel(new Float32Array(y))

  const b1 = new Booster(ds.handle, 'objective=binary verbosity=-1 num_leaves=8')
  for (let i = 0; i < 10; i++) b1.update()
  ds.dispose()

  const modelBytes = b1.saveModel()
  assert(modelBytes.length > 0, 'model bytes should not be empty')

  const b2 = Booster.loadModel(modelBytes)
  const p1 = b1.predict(flat, rows, cols)
  const p2 = b2.predict(flat, rows, cols)

  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-6, `pred[${i}] mismatch`)
  }

  b1.dispose()
  b2.dispose()
})

await test('Booster getNumClasses', async () => {
  const { X, y } = makeMulticlassData(100, 3)
  const rows = X.length, cols = X[0].length
  const flat = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) flat[i * cols + j] = X[i][j]
  }

  const ds = new Dataset(flat, rows, cols, 'objective=multiclass num_class=3 verbosity=-1')
  ds.setLabel(new Float32Array(y))

  const booster = new Booster(ds.handle, 'objective=multiclass num_class=3 verbosity=-1')
  booster.update()

  const nc = booster.getNumClasses()
  assert(nc === 3, `expected 3 classes, got ${nc}`)

  ds.dispose()
  booster.dispose()
})

console.log('\n-- Binary Classification (LGBModel) --')

await test('LGBModel.create returns unfitted model', async () => {
  const model = await LGBModel.create({ objective: 'binary' })
  assert(!model.isFitted, 'should not be fitted')
  model.dispose()
})

await test('Binary classification fit + predict', async () => {
  const { X, y } = makeBinaryData(200)
  const model = await LGBModel.create({
    objective: 'binary',
    learning_rate: 0.1,
    num_leaves: 15,
    numRound: 50
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nrClass === 2, `expected 2 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  assert(preds.length === X.length, 'prediction count should match')

  // Check predictions are class labels
  const unique = new Set(preds)
  assert(unique.has(0) || unique.has(1), 'predictions should be class labels')

  const score = model.score(X, y)
  assert(score > 0.7, `accuracy should be > 0.7, got ${score}`)

  model.dispose()
})

await test('Binary predictProba', async () => {
  const { X, y } = makeBinaryData(100)
  const model = await LGBModel.create({
    objective: 'binary',
    numRound: 30,
    num_leaves: 8
  })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba.length === X.length * 2, `expected ${X.length * 2} proba values, got ${proba.length}`)

  // Each row should sum to ~1
  for (let i = 0; i < X.length; i++) {
    const sum = proba[i * 2] + proba[i * 2 + 1]
    assertClose(sum, 1.0, 1e-6, `row ${i} proba sum`)
  }

  model.dispose()
})

await test('Binary with sparse labels (3, 7)', async () => {
  const { X, y: rawY } = makeBinaryData(100)
  const y = rawY.map(v => v === 0 ? 3 : 7)

  const model = await LGBModel.create({ objective: 'binary', numRound: 30, num_leaves: 8 })
  model.fit(X, y)

  const classes = model.classes
  assert(classes[0] === 3 && classes[1] === 7, `expected classes [3,7], got [${classes}]`)

  const preds = model.predict(X)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 3 || preds[i] === 7, `pred should be 3 or 7, got ${preds[i]}`)
  }

  model.dispose()
})

console.log('\n-- Multiclass Classification --')

await test('Multiclass fit + predict', async () => {
  const { X, y } = makeMulticlassData(200, 3)
  const model = await LGBModel.create({
    objective: 'multiclass',
    num_leaves: 15,
    numRound: 50,
    learning_rate: 0.1
  })
  model.fit(X, y)

  assert(model.nrClass === 3, `expected 3 classes, got ${model.nrClass}`)

  const preds = model.predict(X)
  assert(preds.length === X.length, 'prediction count should match')

  const unique = new Set(preds)
  assert(unique.size <= 3, `expected at most 3 unique preds, got ${unique.size}`)

  const score = model.score(X, y)
  assert(score > 0.5, `accuracy should be > 0.5, got ${score}`)

  model.dispose()
})

await test('Multiclass predictProba', async () => {
  const { X, y } = makeMulticlassData(100, 3)
  const model = await LGBModel.create({
    objective: 'multiclass',
    numRound: 30,
    num_leaves: 8
  })
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba.length === X.length * 3, `expected ${X.length * 3} proba values, got ${proba.length}`)

  // Each row should sum to ~1
  for (let i = 0; i < X.length; i++) {
    const sum = proba[i * 3] + proba[i * 3 + 1] + proba[i * 3 + 2]
    assertClose(sum, 1.0, 1e-5, `row ${i} proba sum`)
  }

  model.dispose()
})

console.log('\n-- Regression --')

await test('Regression fit + predict', async () => {
  const { X, y } = makeRegressionData(200)
  const model = await LGBModel.create({
    objective: 'regression',
    learning_rate: 0.1,
    num_leaves: 31,
    numRound: 100
  })
  model.fit(X, y)

  assert(model.isFitted, 'should be fitted')
  assert(model.nrClass === 0, 'regression should have 0 classes')

  const preds = model.predict(X)
  assert(preds.length === X.length, 'prediction count should match')

  const score = model.score(X, y)
  assert(score > 0.5, `R2 should be > 0.5, got ${score}`)

  model.dispose()
})

await test('Regression predictProba throws', async () => {
  const { X, y } = makeRegressionData(50)
  const model = await LGBModel.create({ objective: 'regression', numRound: 10 })
  model.fit(X, y)

  let threw = false
  try {
    model.predictProba(X)
  } catch (e) {
    threw = true
    assert(e.message.includes('classification'), 'should mention classification')
  }
  assert(threw, 'should throw for regression predictProba')

  model.dispose()
})

console.log('\n-- Save/Load --')

await test('Save and load round-trip (binary)', async () => {
  const { X, y } = makeBinaryData(100)
  const model = await LGBModel.create({ objective: 'binary', numRound: 30, num_leaves: 8 })
  model.fit(X, y)

  const bundle = model.save()
  assert(bundle instanceof Uint8Array, 'bundle should be Uint8Array')
  assert(bundle.length > 100, 'bundle should be non-trivial')

  // Verify bundle format
  const { manifest } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.lightgbm.classifier@1', `typeId should be classifier, got ${manifest.typeId}`)

  const loaded = await LGBModel.load(bundle)
  assert(loaded.isFitted, 'loaded model should be fitted')

  const p1 = model.predict(X)
  const p2 = loaded.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-6, `pred[${i}]`)
  }

  model.dispose()
  loaded.dispose()
})

await test('Save and load round-trip (regression)', async () => {
  const { X, y } = makeRegressionData(100)
  const model = await LGBModel.create({ objective: 'regression', numRound: 30 })
  model.fit(X, y)

  const bundle = model.save()
  const { manifest } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.lightgbm.regressor@1', `typeId should be regressor, got ${manifest.typeId}`)

  const loaded = await LGBModel.load(bundle)
  const p1 = model.predict(X)
  const p2 = loaded.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-6, `pred[${i}]`)
  }

  model.dispose()
  loaded.dispose()
})

await test('Save and load round-trip (multiclass)', async () => {
  const { X, y } = makeMulticlassData(100, 3)
  const model = await LGBModel.create({
    objective: 'multiclass',
    numRound: 30,
    num_leaves: 8
  })
  model.fit(X, y)

  const bundle = model.save()
  const loaded = await LGBModel.load(bundle)

  const p1 = model.predict(X)
  const p2 = loaded.predict(X)
  for (let i = 0; i < p1.length; i++) {
    assertClose(p1[i], p2[i], 1e-6, `pred[${i}]`)
  }

  model.dispose()
  loaded.dispose()
})

console.log('\n-- Error Handling --')

await test('Predict on unfitted model throws NotFittedError', async () => {
  const model = await LGBModel.create({ objective: 'binary' })
  let threw = false
  try {
    model.predict([[1, 2]])
  } catch (e) {
    threw = true
    assert(e.message.includes('not fitted'), 'should mention not fitted')
  }
  assert(threw, 'should throw')
  model.dispose()
})

await test('Predict on disposed model throws DisposedError', async () => {
  const { X, y } = makeBinaryData(50)
  const model = await LGBModel.create({ objective: 'binary', numRound: 5, num_leaves: 4 })
  model.fit(X, y)
  model.dispose()

  let threw = false
  try {
    model.predict(X)
  } catch (e) {
    threw = true
    assert(e.message.includes('disposed'), 'should mention disposed')
  }
  assert(threw, 'should throw')
})

await test('Double dispose is safe', async () => {
  const model = await LGBModel.create({ objective: 'binary' })
  model.dispose()
  model.dispose() // should not throw
})

console.log('\n-- Parameters --')

await test('getParams returns params', async () => {
  const model = await LGBModel.create({ objective: 'binary', learning_rate: 0.05 })
  const params = model.getParams()
  assert(params.objective === 'binary', 'objective should be binary')
  assert(params.learning_rate === 0.05, 'learning_rate should be 0.05')
  model.dispose()
})

await test('setParams updates params', async () => {
  const model = await LGBModel.create({ objective: 'binary' })
  model.setParams({ learning_rate: 0.01 })
  assert(model.getParams().learning_rate === 0.01, 'learning_rate should be updated')
  model.dispose()
})

await test('defaultSearchSpace returns valid space', async () => {
  const space = LGBModel.defaultSearchSpace()
  assert(space.objective, 'should have objective')
  assert(space.objective.type === 'categorical', 'objective should be categorical')
  assert(space.learning_rate, 'should have learning_rate')
  assert(space.num_leaves, 'should have num_leaves')
  assert(space.numRound, 'should have numRound')
})

console.log('\n-- Capabilities --')

await test('Classifier capabilities', async () => {
  const model = await LGBModel.create({ objective: 'binary' })
  const caps = model.capabilities
  assert(caps.classifier === true, 'should be classifier')
  assert(caps.regressor === false, 'should not be regressor')
  assert(caps.predictProba === true, 'should support predictProba')
  model.dispose()
})

await test('Regressor capabilities', async () => {
  const model = await LGBModel.create({ objective: 'regression' })
  const caps = model.capabilities
  assert(caps.classifier === false, 'should not be classifier')
  assert(caps.regressor === true, 'should be regressor')
  assert(caps.predictProba === false, 'should not support predictProba')
  model.dispose()
})

console.log('\n-- Refit --')

await test('Refit replaces previous model', async () => {
  const data1 = makeBinaryData(100, 42)
  const data2 = makeBinaryData(100, 99)
  const model = await LGBModel.create({ objective: 'binary', numRound: 10, num_leaves: 8 })

  model.fit(data1.X, data1.y)
  const p1 = model.predict(data1.X)

  model.fit(data2.X, data2.y)
  const p2 = model.predict(data1.X)

  // Predictions should differ after refit with different data
  let differ = false
  for (let i = 0; i < p1.length; i++) {
    if (p1[i] !== p2[i]) { differ = true; break }
  }
  assert(differ, 'predictions should differ after refit')

  model.dispose()
})

console.log('\n-- Extra Trees --')

await test('Extra trees mode works', async () => {
  const { X, y } = makeBinaryData(100)
  const model = await LGBModel.create({
    objective: 'binary',
    extra_trees: true,
    numRound: 30,
    num_leaves: 15
  })
  model.fit(X, y)

  const score = model.score(X, y)
  assert(score > 0.5, `extra trees accuracy should be > 0.5, got ${score}`)

  model.dispose()
})

// --- Summary ---

console.log(`\n=== Results: ${passed}/${total} passed, ${failed} failed ===\n`)
if (failed > 0) process.exit(1)
