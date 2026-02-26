import { getWasm } from './wasm.js'

// FinalizationRegistry safety net
const registry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/lightgbm: Booster was not disposed -- calling free() automatically.')
      freeFn(ptr[0])
    }
  })
  : null

function withCString(wasm, str, fn) {
  const bytes = new TextEncoder().encode(str + '\0')
  const ptr = wasm._malloc(bytes.length)
  wasm.HEAPU8.set(bytes, ptr)
  try {
    return fn(ptr)
  } finally {
    wasm._free(ptr)
  }
}

function getLastError(wasm) {
  return wasm.UTF8ToString(wasm._wl_lgb_get_last_error())
}

// Internal sentinel for loadModel path
const LOAD_SENTINEL = Symbol('load')

export class Booster {
  #handle = null
  #freed = false
  #ptrRef = null

  constructor(trainDataHandle, paramsStr) {
    // Internal path: loadModel passes sentinel + handle
    if (trainDataHandle === LOAD_SENTINEL) {
      this.#handle = paramsStr // second arg holds the handle
      this.#freed = false
      this.#ptrRef = [this.#handle]
      if (registry) {
        registry.register(this, {
          ptr: this.#ptrRef,
          freeFn: (h) => getWasm()._wl_lgb_booster_free(h)
        }, this)
      }
      return
    }

    const wasm = getWasm()
    const outPtr = wasm._malloc(4)

    const ret = withCString(wasm, paramsStr, (paramsPtr) =>
      wasm._wl_lgb_booster_create(trainDataHandle, paramsPtr, outPtr)
    )

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`Booster creation failed: ${getLastError(wasm)}`)
    }

    this.#handle = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    // Leak detection
    this.#ptrRef = [this.#handle]
    if (registry) {
      registry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_lgb_booster_free(h)
      }, this)
    }
  }

  get handle() {
    if (this.#freed) throw new Error('Booster already disposed')
    return this.#handle
  }

  update() {
    const wasm = getWasm()
    const finishedPtr = wasm._malloc(4)
    wasm.setValue(finishedPtr, 0, 'i32')

    const ret = wasm._wl_lgb_booster_update(this.handle, finishedPtr)
    const finished = wasm.getValue(finishedPtr, 'i32')
    wasm._free(finishedPtr)

    if (ret !== 0) {
      throw new Error(`Booster update failed: ${getLastError(wasm)}`)
    }
    return finished !== 0
  }

  getNumClasses() {
    const wasm = getWasm()
    const outPtr = wasm._malloc(4)

    const ret = wasm._wl_lgb_booster_get_num_classes(this.handle, outPtr)
    const nc = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    if (ret !== 0) {
      throw new Error(`Booster getNumClasses failed: ${getLastError(wasm)}`)
    }
    return nc
  }

  predict(data, nrow, ncol, { predictType = 0, numIteration = 0 } = {}) {
    const wasm = getWasm()

    // Copy float32 data to WASM heap
    const dataPtr = wasm._malloc(data.length * 4)
    wasm.HEAPF32.set(data, dataPtr / 4)

    // Output length
    const outLenPtr = wasm._malloc(4)

    // Allocate output buffer (estimate: nrow * numClasses)
    // For safety, allocate max(nrow * max_classes, nrow * 100)
    const maxOut = nrow * 100
    const outResultPtr = wasm._malloc(maxOut * 8)

    const ret = withCString(wasm, '', (paramPtr) =>
      wasm._wl_lgb_booster_predict(
        this.handle, dataPtr, nrow, ncol,
        predictType, numIteration, paramPtr,
        outLenPtr, outResultPtr
      )
    )

    wasm._free(dataPtr)

    if (ret !== 0) {
      wasm._free(outLenPtr)
      wasm._free(outResultPtr)
      throw new Error(`Booster predict failed: ${getLastError(wasm)}`)
    }

    const outLen = wasm.getValue(outLenPtr, 'i32')
    wasm._free(outLenPtr)

    // Copy results
    const result = new Float64Array(outLen)
    for (let i = 0; i < outLen; i++) {
      result[i] = wasm.HEAPF64[outResultPtr / 8 + i]
    }
    wasm._free(outResultPtr)

    return result
  }

  saveModel() {
    const wasm = getWasm()

    // First pass: get required buffer length
    const outLenPtr = wasm._malloc(4)
    let ret = wasm._wl_lgb_booster_save_model(this.handle, 0, outLenPtr, 0)

    if (ret !== 0) {
      wasm._free(outLenPtr)
      throw new Error(`Booster saveModel (size query) failed: ${getLastError(wasm)}`)
    }

    const bufLen = wasm.getValue(outLenPtr, 'i32')

    // Second pass: get actual model string
    const bufPtr = wasm._malloc(bufLen)
    ret = wasm._wl_lgb_booster_save_model(this.handle, bufLen, outLenPtr, bufPtr)

    if (ret !== 0) {
      wasm._free(outLenPtr)
      wasm._free(bufPtr)
      throw new Error(`Booster saveModel failed: ${getLastError(wasm)}`)
    }

    const actualLen = wasm.getValue(outLenPtr, 'i32')
    wasm._free(outLenPtr)

    // Copy to JS Uint8Array (text model, null-terminated)
    const result = new Uint8Array(actualLen - 1) // exclude null terminator
    result.set(wasm.HEAPU8.subarray(bufPtr, bufPtr + actualLen - 1))
    wasm._free(bufPtr)

    return result
  }

  static loadModel(modelBytes) {
    const wasm = getWasm()

    // Copy model string to WASM heap (add null terminator)
    const buf = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes)
    const bufPtr = wasm._malloc(buf.length + 1)
    wasm.HEAPU8.set(buf, bufPtr)
    wasm.HEAPU8[bufPtr + buf.length] = 0 // null terminator

    const outIterPtr = wasm._malloc(4)
    const outPtr = wasm._malloc(4)

    const ret = wasm._wl_lgb_booster_load_model(bufPtr, outIterPtr, outPtr)

    wasm._free(bufPtr)
    wasm._free(outIterPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`Booster loadModel failed: ${getLastError(wasm)}`)
    }

    const handle = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    return new Booster(LOAD_SENTINEL, handle)
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    const wasm = getWasm()
    wasm._wl_lgb_booster_free(this.#handle)

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (registry) registry.unregister(this)

    this.#handle = null
  }
}
