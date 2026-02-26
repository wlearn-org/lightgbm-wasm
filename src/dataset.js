import { getWasm } from './wasm.js'

// FinalizationRegistry safety net
const registry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/lightgbm: Dataset was not disposed -- calling free() automatically.')
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

export class Dataset {
  #handle = null
  #freed = false
  #ptrRef = null

  constructor(data, nrow, ncol, params = '') {
    const wasm = getWasm()

    // Copy float32 data to WASM heap
    const dataBytes = data.length * 4
    const dataPtr = wasm._malloc(dataBytes)
    wasm.HEAPF32.set(data, dataPtr / 4)

    // Output handle pointer
    const outPtr = wasm._malloc(4)

    const ret = withCString(wasm, params, (paramsPtr) =>
      wasm._wl_lgb_dataset_create_from_mat(dataPtr, nrow, ncol, paramsPtr, outPtr)
    )

    if (ret !== 0) {
      wasm._free(dataPtr)
      wasm._free(outPtr)
      throw new Error(`Dataset creation failed: ${getLastError(wasm)}`)
    }

    this.#handle = wasm.getValue(outPtr, 'i32')
    wasm._free(dataPtr)
    wasm._free(outPtr)

    // Leak detection
    this.#ptrRef = [this.#handle]
    if (registry) {
      registry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getWasm()._wl_lgb_dataset_free(h)
      }, this)
    }
  }

  get handle() {
    if (this.#freed) throw new Error('Dataset already disposed')
    return this.#handle
  }

  setLabel(labels) {
    const wasm = getWasm()
    const arr = labels instanceof Float32Array ? labels : new Float32Array(labels)
    const ptr = wasm._malloc(arr.length * 4)
    wasm.HEAPF32.set(arr, ptr / 4)

    // C_API_DTYPE_FLOAT32 = 0
    const ret = withCString(wasm, 'label', (fieldPtr) =>
      wasm._wl_lgb_dataset_set_field(this.handle, fieldPtr, ptr, arr.length, 0)
    )

    wasm._free(ptr)

    if (ret !== 0) {
      throw new Error(`Dataset setLabel failed: ${getLastError(wasm)}`)
    }
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    const wasm = getWasm()
    wasm._wl_lgb_dataset_free(this.#handle)

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (registry) registry.unregister(this)

    this.#handle = null
  }
}
