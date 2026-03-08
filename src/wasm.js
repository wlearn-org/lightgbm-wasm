// WASM loader -- loads the LightGBM WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadLGB(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createLightGBM = require('../wasm/lightgbm.js')
    wasmModule = await createLightGBM(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadLGB() first')
  return wasmModule
}

module.exports = { loadLGB, getWasm }
