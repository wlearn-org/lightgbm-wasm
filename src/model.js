import { loadLGB, getWasm } from './wasm.js'
import { Dataset } from './dataset.js'
import { Booster } from './booster.js'
import {
  normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ref, freeFn }) => {
    if (ref[0]) {
      console.warn('@wlearn/lightgbm: LGBModel was not disposed -- calling free() automatically.')
      freeFn(ref[0])
    }
  })
  : null

// --- Objective classification ---

const CLASSIFIER_OBJECTIVES = new Set([
  'binary', 'multiclass', 'multiclassova', 'cross_entropy'
])

const PROBA_OBJECTIVES = new Set([
  'binary', 'multiclass', 'multiclassova'
])

// LightGBM params that are wlearn-only (not passed to Booster)
const WLEARN_PARAMS = new Set(['numRound', 'coerce', 'task'])

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- LGBModel ---

export class LGBModel {
  #booster = null
  #freed = false
  #boosterRef = null
  #params = {}
  #fitted = false
  #nrClass = 0
  #classes = null

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      // Load path
      this.#booster = params
      this.#params = extra.params || {}
      this.#nrClass = extra.nrClass || 0
      this.#classes = extra.classes ? new Int32Array(extra.classes) : null
      this.#fitted = true
      this.#freed = false
      this.#boosterRef = [this.#booster]
      if (leakRegistry) {
        leakRegistry.register(this, {
          ref: this.#boosterRef,
          freeFn: (b) => { try { b.dispose() } catch {} }
        }, this)
      }
    } else {
      // Normal construction from create()
      this.#params = handle || {}
      this.#freed = false
    }
  }

  static async create(params = {}) {
    await loadLGB()
    return new LGBModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureNotDisposed()

    // Map task param to objective if needed
    this.#resolveTask(y)

    // Dispose previous booster if refitting
    if (this.#booster) {
      this.#booster.dispose()
      this.#booster = null
      if (this.#boosterRef) this.#boosterRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)

    // Detect objective (default to regression)
    const obj = this.#params.objective || 'regression'

    // For classifiers: validate and extract classes, remap to 0-based
    let yTrain
    if (CLASSIFIER_OBJECTIVES.has(obj)) {
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) {
        const v = yNorm[i]
        if (v !== Math.floor(v)) {
          throw new Error(`Classifier labels must be integers, got ${v} at index ${i}`)
        }
        unique.add(v)
      }
      const sorted = [...unique].sort((a, b) => a - b)
      this.#classes = new Int32Array(sorted)
      this.#nrClass = sorted.length

      // Remap to 0-based contiguous
      const classMap = new Map()
      for (let i = 0; i < sorted.length; i++) classMap.set(sorted[i], i)
      yTrain = new Float32Array(yNorm.length)
      for (let i = 0; i < yNorm.length; i++) yTrain[i] = classMap.get(yNorm[i])
    } else {
      this.#classes = null
      this.#nrClass = 0
      yTrain = yNorm instanceof Float32Array ? yNorm : new Float32Array(yNorm)
    }

    if (yTrain.length !== rows) {
      throw new Error(`y length (${yTrain.length}) does not match X rows (${rows})`)
    }

    // Build LightGBM param string: "key1=value1 key2=value2"
    const numRound = this.#params.numRound || 100
    const lgbParams = {}
    for (const [key, val] of Object.entries(this.#params)) {
      if (!WLEARN_PARAMS.has(key)) lgbParams[key] = val
    }
    // Defaults
    if (!('objective' in lgbParams)) lgbParams.objective = obj
    if (!('verbosity' in lgbParams)) lgbParams.verbosity = -1

    // Auto-set num_class for multiclass
    if ((obj === 'multiclass' || obj === 'multiclassova') &&
        !('num_class' in lgbParams) && this.#nrClass > 0) {
      lgbParams.num_class = this.#nrClass
    }

    const paramStr = Object.entries(lgbParams)
      .map(([k, v]) => `${k}=${v}`)
      .join(' ')

    // Create Dataset
    const ds = new Dataset(xData, rows, cols, paramStr)
    ds.setLabel(yTrain)

    // Create Booster and train
    const booster = new Booster(ds.handle, paramStr)
    for (let i = 0; i < numRound; i++) {
      booster.update()
    }

    ds.dispose()

    this.#booster = booster
    this.#fitted = true

    this.#boosterRef = [this.#booster]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ref: this.#boosterRef,
        freeFn: (b) => { try { b.dispose() } catch {} }
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { data: xData, rows, cols } = this.#normalizeX(X)
    const rawPreds = this.#booster.predict(xData, rows, cols)

    const obj = this.#params.objective || 'regression'

    if (!CLASSIFIER_OBJECTIVES.has(obj)) {
      return rawPreds
    }

    const result = new Float64Array(rows)

    if (obj === 'binary') {
      // Raw is P(class=1), threshold at 0.5
      for (let i = 0; i < rows; i++) {
        result[i] = this.#classes[rawPreds[i] > 0.5 ? 1 : 0]
      }
    } else if (obj === 'multiclass' || obj === 'multiclassova') {
      // Raw is rows * nrClass probabilities, argmax
      const nc = this.#nrClass
      for (let i = 0; i < rows; i++) {
        let best = 0
        for (let c = 1; c < nc; c++) {
          if (rawPreds[i * nc + c] > rawPreds[i * nc + best]) best = c
        }
        result[i] = this.#classes[best]
      }
    } else {
      // cross_entropy: threshold at 0.5
      for (let i = 0; i < rows; i++) {
        result[i] = this.#classes[rawPreds[i] > 0.5 ? 1 : 0]
      }
    }

    return result
  }

  predictProba(X) {
    this.#ensureFitted()
    const obj = this.#params.objective || 'regression'

    if (!PROBA_OBJECTIVES.has(obj)) {
      throw new Error(`predictProba requires classification objective, got "${obj}"`)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const rawPreds = this.#booster.predict(xData, rows, cols)

    if (obj === 'binary') {
      // LightGBM returns P(class=1). Expand to rows * 2: [P(class=0), P(class=1)]
      const result = new Float64Array(rows * 2)
      for (let i = 0; i < rows; i++) {
        const p1 = rawPreds[i]
        result[i * 2] = 1 - p1
        result[i * 2 + 1] = p1
      }
      return result
    }

    // multiclass / multiclassova: already rows * nrClass
    return new Float64Array(rawPreds)
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (!this.#isClassifier()) {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }

    // Accuracy
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === yArr[i]) correct++
    }
    return correct / preds.length
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const modelBytes = this.#booster.saveModel()
    const typeId = this.#isClassifier()
      ? 'wlearn.lightgbm.classifier@1'
      : 'wlearn.lightgbm.regressor@1'
    return encodeBundle(
      {
        typeId,
        params: this.getParams(),
        metadata: {
          nrClass: this.#nrClass,
          classes: this.#classes ? Array.from(this.#classes) : [],
          objective: this.#params.objective || 'regression'
        }
      },
      [{ id: 'model', data: modelBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return LGBModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadLGB()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const booster = Booster.loadModel(raw)

    const meta = manifest.metadata || {}
    return new LGBModel(LOAD_SENTINEL, booster, {
      params: manifest.params || {},
      nrClass: meta.nrClass || 0,
      classes: meta.classes || null
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#booster) {
      this.#booster.dispose()
    }

    if (this.#boosterRef) this.#boosterRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#booster = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      objective: { type: 'categorical', values: ['binary', 'regression'] },
      max_depth: { type: 'int_uniform', low: 3, high: 12 },
      learning_rate: { type: 'log_uniform', low: 0.01, high: 0.3 },
      numRound: { type: 'int_uniform', low: 50, high: 500 },
      subsample: { type: 'uniform', low: 0.5, high: 1.0 },
      colsample_bytree: { type: 'uniform', low: 0.5, high: 1.0 },
      min_child_weight: { type: 'log_uniform', low: 1, high: 10 },
      reg_lambda: { type: 'log_uniform', low: 1e-3, high: 10 },
      reg_alpha: { type: 'log_uniform', low: 1e-3, high: 10 },
      num_leaves: { type: 'int_uniform', low: 15, high: 127 }
    }
  }

  // --- Inspection ---

  get nrClass() {
    return this.#nrClass
  }

  get classes() {
    return this.#classes ? Int32Array.from(this.#classes) : new Int32Array(0)
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const obj = this.#params.objective || 'regression'
    const isCls = CLASSIFIER_OBJECTIVES.has(obj)
    return {
      classifier: isCls,
      regressor: !isCls,
      predictProba: PROBA_OBJECTIVES.has(obj),
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get probaDim() {
    if (!this.isFitted) return 0
    const obj = this.#params.objective || 'regression'
    if (obj === 'binary') return 2
    if (obj === 'multiclass' || obj === 'multiclassova') return this.#nrClass
    return 0
  }

  // --- Private helpers ---

  #normalizeX(X) {
    // Fast path: typed matrix { data, rows, cols }
    if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
      const { data, rows, cols } = X
      if (data instanceof Float32Array) return { data, rows, cols }
      return { data: new Float32Array(data), rows, cols }
    }

    // Slow path: number[][]
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { data, rows, cols }
    }

    throw new Error('X must be number[][] or { data: TypedArray, rows, cols }')
  }

  #ensureNotDisposed() {
    if (this.#freed) throw new DisposedError('LGBModel has been disposed.')
  }

  #ensureFitted() {
    if (this.#freed) throw new DisposedError('LGBModel has been disposed.')
    if (!this.#fitted) throw new NotFittedError('LGBModel is not fitted. Call fit() first.')
  }

  #resolveTask(y) {
    const task = this.#params.task
    if (!task) return
    if (this.#params.objective) {
      throw new Error("Cannot set both 'task' and 'objective'. Use one or the other.")
    }
    if (task === 'classification') {
      // Count unique values in y to decide binary vs multiclass
      const yNorm = normalizeY(y)
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) unique.add(yNorm[i])
      if (unique.size > 2) {
        this.#params.objective = 'multiclass'
        this.#params.num_class = unique.size
      } else {
        this.#params.objective = 'binary'
      }
    } else if (task === 'regression') {
      this.#params.objective = 'regression'
    } else {
      throw new Error(`Unknown task: '${task}'. Use 'classification' or 'regression'.`)
    }
  }

  #isClassifier() {
    const obj = this.#params.objective || 'regression'
    return CLASSIFIER_OBJECTIVES.has(obj)
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.lightgbm.classifier@1', async (m, t, b) => LGBModel._fromBundle(m, t, b))
register('wlearn.lightgbm.regressor@1', async (m, t, b) => LGBModel._fromBundle(m, t, b))
