const { LGBModel: LGBModelImpl } = require('./model.js')
const { loadLGB } = require('./wasm.js')
const { Dataset } = require('./dataset.js')
const { Booster } = require('./booster.js')
const { createModelClass } = require('@wlearn/core')

const LGBModel = createModelClass(LGBModelImpl, LGBModelImpl, { name: 'LGBModel', load: loadLGB })

module.exports = { LGBModel, loadLGB, Dataset, Booster }
