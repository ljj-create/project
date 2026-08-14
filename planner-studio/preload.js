const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('planner', {
  todos: {
    get: (dateKey) => ipcRenderer.invoke('todos:get', dateKey),
    all: () => ipcRenderer.invoke('todos:all'),
    add: (dateKey, text, extra) => ipcRenderer.invoke('todos:add', dateKey, text, extra),
    toggle: (dateKey, id) => ipcRenderer.invoke('todos:toggle', dateKey, id),
    remove: (dateKey, id) => ipcRenderer.invoke('todos:remove', dateKey, id),
    rename: (dateKey, id, text) => ipcRenderer.invoke('todos:rename', dateKey, id, text),
    updateRepeat: (templateId, repeat) => ipcRenderer.invoke('todos:updateRepeat', templateId, repeat),
    deleteRepeat: (templateId) => ipcRenderer.invoke('todos:deleteRepeat', templateId),
  },
  tasks: {
    list: () => ipcRenderer.invoke('tasks:list'),
    add: (title, dueAt) => ipcRenderer.invoke('tasks:add', title, dueAt),
    remove: (id) => ipcRenderer.invoke('tasks:remove', id),
    clearOverdue: () => ipcRenderer.invoke('tasks:clearOverdue'),
  },
  settings: {
    get: () => ipcRenderer.invoke('settings:get'),
    set: (partial) => ipcRenderer.invoke('settings:set', partial),
  },
  autoLaunch: (enabled) => ipcRenderer.invoke('app:autoLaunch', enabled),
  backup: {
    export: () => ipcRenderer.invoke('backup:export'),
    import: () => ipcRenderer.invoke('backup:import'),
  },
  close: () => ipcRenderer.send('window:close'),
  openManager: () => ipcRenderer.send('manager:open'),
  onDataChanged: (cb) => ipcRenderer.on('data-changed', () => cb()),
})
